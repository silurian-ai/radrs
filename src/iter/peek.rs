//! Lightweight volume metadata extraction
//!
//! This module provides fast header-only parsing of NEXRAD volumes
//! to extract metadata without downloading/parsing the full file.

use crate::error::{RadrsError, Result};
use crate::fetch::{FETCH_SEMAPHORE, RUNTIME, store_for_bucket};
use chrono::{DateTime, Utc};
use nexrad_data::volume::{Header, Record};
use nexrad_decode::messages::MessageContents;
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, GetRange, ObjectStore};
use pyo3::prelude::*;
use pyo3::types::PyDateTime;
use std::sync::Arc;
use zerocopy::Ref;

/// Volume header size in bytes
const VOLUME_HEADER_SIZE: usize = std::mem::size_of::<Header>();

/// Initial scan size for fast peek (favor fewer round-trips)
const PEEK_FAST_INITIAL: usize = 256 * 1024;

/// Maximum scan size for peek (avoid large S3 reads)
const PEEK_SCAN_MAX: usize = 2 * 1024 * 1024;

/// Maximum number of records to scan while peeking
const PEEK_MAX_RECORDS: usize = 8;

/// Metadata extracted from a NEXRAD volume without full parsing
#[pyclass]
#[derive(Clone, Debug)]
pub struct VolumeMeta {
    /// Radar site ICAO code (e.g., "KTLX")
    #[pyo3(get)]
    pub site: String,

    /// Volume timestamp
    volume_datetime: DateTime<Utc>,

    /// Archive version (e.g., "V06", "V07")
    #[pyo3(get)]
    pub version: String,

    /// File size in bytes (if known from S3 listing)
    #[pyo3(get)]
    pub file_size: Option<u64>,

    // Fields below require decompressing first record (header_only=False)
    /// Volume Coverage Pattern number (e.g., 215, 21, 31)
    #[pyo3(get)]
    pub vcp: Option<u16>,

    /// Radar latitude in degrees
    #[pyo3(get)]
    pub latitude: Option<f32>,

    /// Radar longitude in degrees
    #[pyo3(get)]
    pub longitude: Option<f32>,

    /// Radar altitude in meters
    #[pyo3(get)]
    pub altitude: Option<f32>,

    /// First elevation angle in degrees
    #[pyo3(get)]
    pub first_elevation: Option<f32>,

    /// Moments present in first radial
    #[pyo3(get)]
    pub moments: Option<Vec<String>>,
}

#[pymethods]
impl VolumeMeta {
    fn __repr__(&self) -> String {
        let vcp_str = self
            .vcp
            .map(|v| format!("vcp={}", v))
            .unwrap_or_else(|| "vcp=None".to_string());
        format!(
            "VolumeMeta(site='{}', datetime='{}', version='{}', {})",
            self.site,
            self.volume_datetime.format("%Y-%m-%d %H:%M:%S UTC"),
            self.version,
            vcp_str
        )
    }

    /// Get volume datetime as Python datetime
    #[getter]
    fn datetime<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDateTime>> {
        let timestamp = self.volume_datetime.timestamp();
        let microseconds = self.volume_datetime.timestamp_subsec_micros();
        PyDateTime::from_timestamp(
            py,
            timestamp as f64 + microseconds as f64 / 1_000_000.0,
            None,
        )
    }

    /// Check if this is a clear-air VCP (typically less interesting for ML)
    #[getter]
    fn is_clear_air(&self) -> bool {
        matches!(self.vcp, Some(31) | Some(32) | Some(35))
    }

    /// Check if this is a precipitation/storm VCP
    #[getter]
    fn is_precipitation(&self) -> bool {
        matches!(
            self.vcp,
            Some(12) | Some(21) | Some(112) | Some(121) | Some(212) | Some(215) | Some(221)
        )
    }
}

/// Parse volume header (24 bytes) to extract basic metadata
fn parse_volume_header(data: &[u8]) -> Result<(String, DateTime<Utc>, String)> {
    if data.len() < VOLUME_HEADER_SIZE {
        return Err(RadrsError::Parse(format!(
            "Data too short for volume header: {} bytes",
            data.len()
        )));
    }

    // Use the same approach as nexrad-data's File::header()
    let (header_ref, _rest) = Ref::<_, Header>::from_prefix(data)
        .map_err(|e| RadrsError::Parse(format!("Failed to parse volume header: {:?}", e)))?;
    let header: &Header = Ref::into_ref(header_ref);

    let site = header
        .icao_of_radar()
        .ok_or_else(|| RadrsError::Parse("Invalid ICAO in header".to_string()))?;

    let datetime = header
        .date_time()
        .ok_or_else(|| RadrsError::Parse("Invalid datetime in header".to_string()))?;

    // Extract version from tape_filename (e.g., "AR2V0006." -> "V06")
    let version = header
        .tape_filename()
        .and_then(|s: String| {
            // Format is "AR2V00xx." where xx is the version
            if s.len() >= 8 {
                Some(format!("V{}", &s[6..8]))
            } else {
                None
            }
        })
        .unwrap_or_else(|| "V??".to_string());

    Ok((site, datetime, version))
}

#[derive(Default, Debug)]
struct PeekExtras {
    vcp: Option<u16>,
    latitude: Option<f32>,
    longitude: Option<f32>,
    altitude: Option<f32>,
    first_elevation: Option<f32>,
    moments: Option<Vec<String>>,
}

#[derive(Clone, Copy)]
enum PeekMode {
    Fast,
    Small,
}

impl PeekMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "fast" => Ok(PeekMode::Fast),
            "small" => Ok(PeekMode::Small),
            other => Err(RadrsError::Parse(format!(
                "Invalid peek_mode: {other} (expected 'fast' or 'small')"
            ))),
        }
    }
}

fn scan_record_for_extras(
    record: &Record<'_>,
    record_index: usize,
    extras: &mut PeekExtras,
) -> bool {
    let decompressed = if record.compressed() {
        match record.decompress() {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("Failed to decompress record {}: {}", record_index, e);
                return false;
            }
        }
    } else {
        Record::from_slice(record.data())
    };

    let messages = match decompressed.messages() {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(
                "Failed to decode messages from record {}: {}",
                record_index,
                e
            );
            return false;
        }
    };

    for msg in messages {
        match msg.into_contents() {
            MessageContents::VolumeCoveragePattern(vcp_msg) => {
                extras.vcp = Some(vcp_msg.header().pattern_number() as u16);
            }
            MessageContents::DigitalRadarData(drd) => {
                if let Some(vol_block) = drd.volume_data_block() {
                    if extras.vcp.is_none() {
                        extras.vcp = Some(vol_block.volume_coverage_pattern_number());
                    }
                    if extras.latitude.is_none() {
                        extras.latitude = Some(vol_block.latitude_raw());
                    }
                    if extras.longitude.is_none() {
                        extras.longitude = Some(vol_block.longitude_raw());
                    }
                    if extras.altitude.is_none() {
                        extras.altitude = Some(vol_block.site_height_raw() as f32);
                    }
                }

                if extras.first_elevation.is_none() {
                    extras.first_elevation = Some(drd.header().elevation_angle_raw());
                }

                if extras.moments.is_none() {
                    let mut moments = Vec::new();
                    if drd.reflectivity_data_block().is_some() {
                        moments.push("DBZH".to_string());
                    }
                    if drd.velocity_data_block().is_some() {
                        moments.push("VRADH".to_string());
                    }
                    if drd.spectrum_width_data_block().is_some() {
                        moments.push("WRADH".to_string());
                    }
                    if drd.differential_reflectivity_data_block().is_some() {
                        moments.push("ZDR".to_string());
                    }
                    if drd.differential_phase_data_block().is_some() {
                        moments.push("PHIDP".to_string());
                    }
                    if drd.correlation_coefficient_data_block().is_some() {
                        moments.push("RHOHV".to_string());
                    }
                    if !moments.is_empty() {
                        extras.moments = Some(moments);
                    }
                }
            }
            _ => {}
        }

        // Stop early once we know the VCP; other fields are best-effort.
        if extras.vcp.is_some() {
            return true;
        }
    }

    false
}

fn scan_records_buffer(data: &[u8], scan_max: usize) -> PeekExtras {
    let mut extras = PeekExtras::default();
    let mut position = VOLUME_HEADER_SIZE;
    let mut records_scanned = 0usize;
    let data_len = data.len().min(scan_max);

    while records_scanned < PEEK_MAX_RECORDS {
        if position + 4 > data_len {
            break;
        }

        let mut size_bytes = [0u8; 4];
        size_bytes.copy_from_slice(&data[position..position + 4]);
        let record_size = i32::from_be_bytes(size_bytes).unsigned_abs() as usize;
        let record_end = position + 4 + record_size;

        if record_end > data_len {
            break;
        }

        let record = Record::from_slice(&data[position..record_end]);
        if scan_record_for_extras(&record, records_scanned, &mut extras) {
            break;
        }

        position = record_end;
        records_scanned += 1;
    }

    extras
}

async fn ensure_len_s3(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    data: &mut Vec<u8>,
    target_len: usize,
) -> Result<()> {
    if data.len() >= target_len {
        return Ok(());
    }

    let start = data.len();
    let end = target_len;
    if start >= end {
        return Ok(());
    }

    let chunk = fetch_range(store, path, start, end).await?;
    data.extend_from_slice(&chunk);
    Ok(())
}

async fn scan_records_s3(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    data: &mut Vec<u8>,
    scan_max: usize,
) -> Result<PeekExtras> {
    let mut extras = PeekExtras::default();
    let mut position = VOLUME_HEADER_SIZE;
    let mut records_scanned = 0usize;

    while records_scanned < PEEK_MAX_RECORDS {
        if position + 4 > scan_max {
            break;
        }

        ensure_len_s3(store, path, data, position + 4).await?;
        if data.len() < position + 4 {
            break;
        }

        let mut size_bytes = [0u8; 4];
        size_bytes.copy_from_slice(&data[position..position + 4]);
        let record_size = i32::from_be_bytes(size_bytes).unsigned_abs() as usize;
        let record_end = position + 4 + record_size;

        if record_end > scan_max {
            break;
        }

        ensure_len_s3(store, path, data, record_end).await?;
        if data.len() < record_end {
            break;
        }

        let record = Record::from_slice(&data[position..record_end]);
        if scan_record_for_extras(&record, records_scanned, &mut extras) {
            break;
        }

        position = record_end;
        records_scanned += 1;
    }

    Ok(extras)
}

fn ensure_len_local(file: &mut std::fs::File, data: &mut Vec<u8>, target_len: usize) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    if data.len() >= target_len {
        return Ok(());
    }

    let start = data.len();
    let end = target_len;
    if start >= end {
        return Ok(());
    }

    file.seek(SeekFrom::Start(start as u64))
        .map_err(RadrsError::Io)?;
    let mut chunk = vec![0u8; end - start];
    file.read_exact(&mut chunk).map_err(RadrsError::Io)?;
    data.extend_from_slice(&chunk);
    Ok(())
}

fn scan_records_local(
    file: &mut std::fs::File,
    data: &mut Vec<u8>,
    scan_max: usize,
) -> Result<PeekExtras> {
    let mut extras = PeekExtras::default();
    let mut position = VOLUME_HEADER_SIZE;
    let mut records_scanned = 0usize;

    while records_scanned < PEEK_MAX_RECORDS {
        if position + 4 > scan_max {
            break;
        }

        ensure_len_local(file, data, position + 4)?;
        if data.len() < position + 4 {
            break;
        }

        let mut size_bytes = [0u8; 4];
        size_bytes.copy_from_slice(&data[position..position + 4]);
        let record_size = i32::from_be_bytes(size_bytes).unsigned_abs() as usize;
        let record_end = position + 4 + record_size;

        if record_end > scan_max {
            break;
        }

        ensure_len_local(file, data, record_end)?;
        if data.len() < record_end {
            break;
        }

        let record = Record::from_slice(&data[position..record_end]);
        if scan_record_for_extras(&record, records_scanned, &mut extras) {
            break;
        }

        position = record_end;
        records_scanned += 1;
    }

    Ok(extras)
}

/// Fetch partial data from S3 (range request)
async fn fetch_range(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    let _permit = FETCH_SEMAPHORE.acquire().await.expect("semaphore closed");

    let options = GetOptions {
        range: Some(GetRange::Bounded(start..end)),
        ..Default::default()
    };

    let result = store.get_opts(path, options).await?;
    let bytes = result.bytes().await?;
    Ok(bytes.to_vec())
}

/// Parse S3 URL into (bucket, key)
fn parse_s3_url(url: &str) -> Result<(String, String)> {
    let url = url
        .strip_prefix("s3://")
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Not an S3 URL: {}", url)))?;

    let (bucket, key) = url
        .split_once('/')
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Invalid S3 URL format: s3://{}", url)))?;

    Ok((bucket.to_string(), key.to_string()))
}

/// Peek at a NEXRAD volume to extract metadata without full parsing
///
/// # Arguments
/// * `source` - S3 URL (s3://bucket/path) or local file path
/// * `header_only` - If true, only fetch 24 bytes for basic metadata.
///                   If false (default), adaptively fetch to get VCP, lat/lon, moments.
///
/// # Returns
/// VolumeMeta with extracted fields
pub async fn peek_volume(source: &str, header_only: bool) -> Result<VolumeMeta> {
    peek_volume_with_mode(source, header_only, "fast").await
}

pub async fn peek_volume_with_mode(
    source: &str,
    header_only: bool,
    peek_mode: &str,
) -> Result<VolumeMeta> {
    let mode = PeekMode::parse(peek_mode)?;
    if source.starts_with("s3://") {
        peek_volume_s3(source, header_only, mode).await
    } else {
        peek_volume_local(source, header_only, mode).await
    }
}

async fn peek_volume_s3(source: &str, header_only: bool, mode: PeekMode) -> Result<VolumeMeta> {
    let (bucket, key) = parse_s3_url(source)?;
    let store = store_for_bucket(&bucket)?;
    let path = ObjectPath::from(key);

    // Get file size from HEAD request
    let meta = store.head(&path).await?;
    let file_size = Some(meta.size as u64);

    // If header_only, just fetch the header
    if header_only {
        let data = fetch_range(&store, &path, 0, VOLUME_HEADER_SIZE).await?;
        let (site, datetime, version) = parse_volume_header(&data)?;
        return Ok(VolumeMeta {
            site,
            volume_datetime: datetime,
            version,
            file_size,
            vcp: None,
            latitude: None,
            longitude: None,
            altitude: None,
            first_elevation: None,
            moments: None,
        });
    }

    let file_size_usize = meta.size as usize;
    let scan_max = std::cmp::min(PEEK_SCAN_MAX, file_size_usize).max(VOLUME_HEADER_SIZE);
    let mut data = Vec::new();
    let initial_target = if header_only {
        VOLUME_HEADER_SIZE
    } else {
        match mode {
            PeekMode::Fast => std::cmp::min(PEEK_FAST_INITIAL, scan_max),
            PeekMode::Small => std::cmp::min(VOLUME_HEADER_SIZE + 4, scan_max),
        }
    };
    ensure_len_s3(&store, &path, &mut data, initial_target).await?;
    let (site, datetime, version) = parse_volume_header(&data)?;

    if header_only {
        return Ok(VolumeMeta {
            site,
            volume_datetime: datetime,
            version,
            file_size,
            vcp: None,
            latitude: None,
            longitude: None,
            altitude: None,
            first_elevation: None,
            moments: None,
        });
    }

    let extras = match mode {
        PeekMode::Fast => {
            let mut extras = scan_records_buffer(&data, scan_max);
            if extras.vcp.is_none() && data.len() < scan_max {
                ensure_len_s3(&store, &path, &mut data, scan_max).await?;
                extras = scan_records_buffer(&data, scan_max);
            }
            extras
        }
        PeekMode::Small => scan_records_s3(&store, &path, &mut data, scan_max).await?,
    };

    Ok(VolumeMeta {
        site,
        volume_datetime: datetime,
        version,
        file_size,
        vcp: extras.vcp,
        latitude: extras.latitude,
        longitude: extras.longitude,
        altitude: extras.altitude,
        first_elevation: extras.first_elevation,
        moments: extras.moments,
    })
}

async fn peek_volume_local(source: &str, header_only: bool, mode: PeekMode) -> Result<VolumeMeta> {
    use std::fs::File;
    let mut file = File::open(source).map_err(RadrsError::Io)?;
    let file_size = file.metadata().map(|m| m.len()).ok();
    let file_size_usize = file_size
        .and_then(|s| usize::try_from(s).ok())
        .unwrap_or(PEEK_SCAN_MAX);
    let scan_max = std::cmp::min(PEEK_SCAN_MAX, file_size_usize).max(VOLUME_HEADER_SIZE);
    let mut data = Vec::new();
    let initial_target = if header_only {
        VOLUME_HEADER_SIZE
    } else {
        match mode {
            PeekMode::Fast => std::cmp::min(PEEK_FAST_INITIAL, scan_max),
            PeekMode::Small => std::cmp::min(VOLUME_HEADER_SIZE + 4, scan_max),
        }
    };
    ensure_len_local(&mut file, &mut data, initial_target)?;
    let (site, datetime, version) = parse_volume_header(&data)?;

    if header_only {
        return Ok(VolumeMeta {
            site,
            volume_datetime: datetime,
            version,
            file_size,
            vcp: None,
            latitude: None,
            longitude: None,
            altitude: None,
            first_elevation: None,
            moments: None,
        });
    }

    let extras = match mode {
        PeekMode::Fast => {
            let mut extras = scan_records_buffer(&data, scan_max);
            if extras.vcp.is_none() && data.len() < scan_max {
                ensure_len_local(&mut file, &mut data, scan_max)?;
                extras = scan_records_buffer(&data, scan_max);
            }
            extras
        }
        PeekMode::Small => scan_records_local(&mut file, &mut data, scan_max)?,
    };

    Ok(VolumeMeta {
        site,
        volume_datetime: datetime,
        version,
        file_size,
        vcp: extras.vcp,
        latitude: extras.latitude,
        longitude: extras.longitude,
        altitude: extras.altitude,
        first_elevation: extras.first_elevation,
        moments: extras.moments,
    })
}

/// Peek at a NEXRAD volume to extract metadata without full parsing
///
/// # Arguments
/// * `source` - S3 URL (s3://bucket/path) or local file path
/// * `header_only` - If true, only fetch 24 bytes for basic metadata (site, datetime, version).
///                   If false (default), scan records until VCP is found (up to 2MB cap).
/// * `peek_mode` - "fast" (default) uses larger reads for fewer requests.
///                 "small" minimizes bytes by fetching record-by-record.
///
/// # Example
/// ```python
/// import radrs
///
/// # Quick peek - just header (24 bytes)
/// meta = radrs.peek_volume("s3://unidata-nexrad-level2/2024/07/02/KTLX/KTLX20240702_000556_V06", header_only=True)
/// print(meta.site, meta.datetime, meta.version)
///
/// # Full peek - includes VCP, lat/lon, moments (stops once VCP is found, up to 2MB)
/// meta = radrs.peek_volume("s3://unidata-nexrad-level2/2024/07/02/KTLX/KTLX20240702_000556_V06")
/// print(meta.vcp, meta.is_clear_air, meta.moments)
/// ```
#[pyfunction]
#[pyo3(name = "peek_volume", signature = (source, header_only = false, peek_mode = "fast"))]
pub fn peek_volume_py(
    _py: Python<'_>,
    source: &str,
    header_only: bool,
    peek_mode: &str,
) -> PyResult<VolumeMeta> {
    let meta = if peek_mode == "fast" {
        RUNTIME.block_on(peek_volume(source, header_only))?
    } else {
        RUNTIME.block_on(peek_volume_with_mode(source, header_only, peek_mode))?
    };
    Ok(meta)
}
