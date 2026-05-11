//! Async streaming iterators for NEXRAD Level 2 Archive
//!
//! Provides efficient async iterators that understand the NEXRAD L2 Archive directory structure
//! (YYYY/MM/DD/SITE/) commonly used by AWS S3 (noaa-nexrad-level2), GCS, and other cloud providers.
//!
//! Iterators are initialized with start/end **times** (not dates), and only return volumes whose
//! timestamps fall within that range. The date-based directory structure is an implementation
//! detail for efficient scanning.
//!
//! Supports S3, GCS, Azure, and local filesystem URIs with configurable storage options.

use crate::error::{RadrsError, Result};
use crate::fetch::build_store_from_uri;
use crate::fetch::extract_base_path;
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use futures::stream::{self, StreamExt};
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use pyo3::prelude::*;
use pyo3::types::PyDateTime;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;

/// Metadata extracted from a NEXRAD L2 filename
#[pyclass(skip_from_py_object, module = "radrs._radrs")]
#[derive(Clone, Debug)]
pub struct NexradL2ArchiveInfo {
    /// Full URI to the volume file
    #[pyo3(get)]
    pub uri: String,

    /// Site/instrument name (e.g., "KTLX")
    #[pyo3(get)]
    pub instrument_name: String,

    /// Volume collection timestamp
    vcp_time: DateTime<Utc>,

    /// File size in bytes (if available)
    #[pyo3(get)]
    pub size: Option<u64>,

    /// ObjectStore reference for fetching (not exposed to Python)
    store: Arc<dyn ObjectStore>,

    /// Object path within the store (not exposed to Python)
    object_path: String,
}

#[pymethods]
impl NexradL2ArchiveInfo {
    /// Get vcp_time as Python datetime
    #[getter]
    fn vcp_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDateTime>> {
        let timestamp = self.vcp_time.timestamp();
        let microseconds = self.vcp_time.timestamp_subsec_micros();
        PyDateTime::from_timestamp(
            py,
            timestamp as f64 + microseconds as f64 / 1_000_000.0,
            None,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "NexradL2ArchiveInfo(instrument='{}', vcp_time='{}', uri='{}')",
            self.instrument_name,
            self.vcp_time.format("%Y-%m-%d %H:%M:%S UTC"),
            self.uri
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{}_{}",
            self.instrument_name,
            self.vcp_time.format("%Y%m%d_%H%M%S")
        )
    }
}

impl NexradL2ArchiveInfo {
    /// Fetch the volume file bytes from object storage
    ///
    /// # Arguments
    /// * `max_bytes` - Optional maximum number of bytes to fetch from the start of the file.
    ///   If None, fetches the entire file.
    pub async fn fetch(&self, max_bytes: usize) -> Result<Vec<u8>> {
        let (bytes, _size) = self.fetch_with_size(max_bytes).await?;
        Ok(bytes)
    }

    /// Fetch the volume file bytes from object storage along with the total file size
    ///
    /// # Arguments
    /// * `max_bytes` - Optional maximum number of bytes to fetch from the start of the file.
    ///   If 0, fetches the entire file.
    ///
    /// # Returns
    /// A tuple of (data, total_file_size) where total_file_size is the complete file size in bytes
    pub async fn fetch_with_size(&self, max_bytes: usize) -> Result<(Vec<u8>, u64)> {
        let path = ObjectPath::from(self.object_path.clone());

        let (bytes, total_size) = if max_bytes > 0 {
            // Fetch only the first `max_bytes` bytes
            let range = 0u64..(max_bytes as u64);
            let get_result = self.store.get_range(&path, range).await?;

            // Get total size from metadata if available, otherwise from HEAD request
            let total_size = if let Some(size) = self.size {
                size
            } else {
                let meta = self.store.head(&path).await?;
                meta.size as u64
            };

            (get_result, total_size)
        } else {
            // Fetch the entire file
            let get_result = self.store.get(&path).await?;
            let total_size = get_result.meta.size as u64;
            let bytes = get_result.bytes().await?;
            (bytes, total_size)
        };

        Ok((bytes.to_vec(), total_size))
    }
}

/// Parse Python datetime to UTC DateTime
fn parse_py_datetime(dt: &Bound<'_, PyDateTime>) -> PyResult<DateTime<Utc>> {
    let ts = dt.call_method0("timestamp")?.extract::<f64>()?;
    let seconds = ts as i64;
    let nanos = ((ts - seconds as f64) * 1_000_000_000.0) as u32;

    DateTime::from_timestamp(seconds, nanos)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid timestamp"))
}

/// Parse NEXRAD L2 filename to extract metadata
///
/// Expected format: {SITE}{YYYYMMDD}_{HHMMSS}_V##
/// Example: KTLX20240315_120000_V06
fn parse_l2_filename(filename: &str) -> Result<(String, DateTime<Utc>)> {
    // Pattern: 4 uppercase alphanumeric chars, 8 digits (date), underscore, 6 digits (time), _V, 2 digits
    let re = Regex::new(r"^([A-Z0-9]{4})(\d{8})_(\d{6})_V\d{2}$")
        .map_err(|e| RadrsError::Parse(format!("Invalid regex: {}", e)))?;

    let caps = re
        .captures(filename)
        .ok_or_else(|| RadrsError::Parse(format!("Invalid L2 filename: {}", filename)))?;

    let instrument_name = caps.get(1).unwrap().as_str().to_string();
    let date_str = caps.get(2).unwrap().as_str();
    let time_str = caps.get(3).unwrap().as_str();

    // Parse datetime
    let datetime_str = format!("{}{}", date_str, time_str);
    let naive_dt = chrono::NaiveDateTime::parse_from_str(&datetime_str, "%Y%m%d%H%M%S")
        .map_err(|e| RadrsError::Parse(format!("Invalid datetime in filename: {}", e)))?;

    let vcp_time = DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc);

    Ok((instrument_name, vcp_time))
}

/// Configuration for L2 volume iteration
pub struct NexradL2ArchiveIterConfig {
    /// Base URI (e.g., "s3://noaa-nexrad-level2", "gs://bucket", "/local/path")
    pub base_uri: String,

    /// Start time (inclusive) - only volumes with timestamps >= this will be returned
    pub start_time: DateTime<Utc>,

    /// End time (inclusive) - only volumes with timestamps <= this will be returned
    pub end_time: DateTime<Utc>,

    /// Optional storage options (credentials, region, etc.)
    /// Examples:
    /// - S3: {"region": "us-east-1", "access_key_id": "...", "secret_access_key": "..."}
    /// - GCS: {"service_account_path": "/path/to/key.json"}
    /// - Azure: {"account_name": "...", "access_key": "..."}
    pub storage_options: Option<HashMap<String, String>>,

    /// Optional site filter (e.g., vec!["KTLX", "KFWS"])
    pub site_filter: Option<Vec<String>>,

    /// Maximum number of concurrent site directory listings (default: 10)
    pub max_concurrent_ls: usize,
}

/// Async iterator state for L2 volumes
#[derive(Clone)]
pub struct NexradL2ArchiveIterator {
    store: Arc<dyn ObjectStore>,
    base_path: String,
    current_date: NaiveDate,
    end_date: NaiveDate,
    site_filter: Option<Vec<String>>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    pending_volumes: Vec<NexradL2ArchiveInfo>,
    index: usize,
    max_concurrent_ls: usize,
}

impl NexradL2ArchiveIterator {
    /// Create a new L2 volume iterator
    ///
    /// Supports S3, GCS, Azure, and local filesystem URIs with configurable storage_options.
    /// The iterator only returns volumes whose timestamps fall within [start_time, end_time].
    ///
    /// # Examples
    ///
    /// S3 (anonymous):
    /// ```ignore
    /// use chrono::{DateTime, Utc};
    /// let config = NexradL2ArchiveIterConfig {
    ///     base_uri: "s3://noaa-nexrad-level2".to_string(),
    ///     start_time: "2024-03-15T10:00:00Z".parse::<DateTime<Utc>>().unwrap(),
    ///     end_time: "2024-03-15T14:00:00Z".parse::<DateTime<Utc>>().unwrap(),
    ///     storage_options: Some(hashmap!{"anon" => "true"}),
    ///     site_filter: Some(vec!["KTLX".to_string()]),
    /// };
    /// ```
    ///
    /// GCS with credentials:
    /// ```ignore
    /// let config = NexradL2ArchiveIterConfig {
    ///     base_uri: "gs://my-bucket/nexrad".to_string(),
    ///     start_time: "2024-03-15T00:00:00Z".parse().unwrap(),
    ///     end_time: "2024-03-16T00:00:00Z".parse().unwrap(),
    ///     storage_options: Some(hashmap!{"service_account_path" => "/path/to/key.json"}),
    ///     site_filter: None,
    /// };
    /// ```
    ///
    /// Local filesystem:
    /// ```ignore
    /// let config = NexradL2ArchiveIterConfig {
    ///     base_uri: "/data/nexrad".to_string(),
    ///     start_time: "2024-03-15T00:00:00Z".parse().unwrap(),
    ///     end_time: "2024-03-15T23:59:59Z".parse().unwrap(),
    ///     storage_options: None,
    ///     site_filter: None,
    /// };
    /// ```
    pub fn new(config: NexradL2ArchiveIterConfig) -> Result<Self> {
        // Build ObjectStore from URI with storage options
        let store = build_store_from_uri(&config.base_uri, config.storage_options)?;

        // Extract base path from URI (remove scheme and bucket/container)
        let base_path = extract_base_path(&config.base_uri)?;

        // Compute date range from time bounds (dates are just for scanning directories)
        let start_date = config.start_time.date_naive();
        let end_date = config.end_time.date_naive();

        Ok(Self {
            store,
            base_path,
            current_date: start_date,
            end_date,
            site_filter: config.site_filter,
            start_time: config.start_time,
            end_time: config.end_time,
            pending_volumes: Vec::new(),
            index: 0,
            max_concurrent_ls: config.max_concurrent_ls,
        })
    }

    /// Get next volume info
    pub async fn next(&mut self) -> Result<Option<NexradL2ArchiveInfo>> {
        loop {
            // Return from pending volumes if available
            if self.index < self.pending_volumes.len() {
                let info = self.pending_volumes[self.index].clone();
                self.index += 1;
                return Ok(Some(info));
            }

            // Move to next date if we've exhausted current date
            if self.current_date > self.end_date {
                return Ok(None);
            }

            // List volumes for current date
            self.pending_volumes = self.list_volumes_for_date(self.current_date).await?;
            self.index = 0;

            // Move to next date
            self.current_date = self.current_date.succ_opt().unwrap_or(self.current_date);
        }
    }

    /// List all volumes for a specific date
    async fn list_volumes_for_date(&self, date: NaiveDate) -> Result<Vec<NexradL2ArchiveInfo>> {
        // Build date path: YYYY/MM/DD
        let date_path = format!(
            "{}/{:04}/{:02}/{:02}",
            self.base_path,
            date.year(),
            date.month(),
            date.day()
        );
        let date_path_obj = ObjectPath::from(date_path.clone());

        // List site directories
        let mut site_dirs = Vec::new();

        if let Some(ref filter) = self.site_filter {
            for site in filter {
                site_dirs.push(ObjectPath::from(format!("{}/{}", date_path, site)));
            }
        } else {
            // Use list_with_delimiter for non-recursive listing (more efficient)
            // This returns common_prefixes which are the site directories
            let result = self.store.list_with_delimiter(Some(&date_path_obj)).await;

            if let Ok(list_result) = result {
                // common_prefixes contains site directory paths like "2024/03/15/KTLX/"
                // objects would contain files directly in the date directory (shouldn't exist in well-structured archives)
                site_dirs = list_result.common_prefixes;
            }
        }

        // List volumes in each site directory (in parallel)
        let volumes_futures = site_dirs.into_iter().map(|site_dir| {
            let store = self.store.clone();
            let start_time = self.start_time;
            let end_time = self.end_time;

            async move { list_volumes_in_site(store, site_dir, start_time, end_time).await }
        });

        // Collect all volumes
        let volumes_results: Vec<Result<Vec<NexradL2ArchiveInfo>>> = stream::iter(volumes_futures)
            .buffer_unordered(self.max_concurrent_ls) // Process N sites concurrently
            .collect()
            .await;

        let mut all_volumes = Vec::new();
        for result in volumes_results {
            match result {
                Ok(mut volumes) => all_volumes.append(&mut volumes),
                Err(_) => continue,
            }
        }

        // Sort by vcp_time
        all_volumes.sort_by_key(|v| v.vcp_time);

        Ok(all_volumes)
    }
}

/// List volumes within a specific site directory, filtering by time range
async fn list_volumes_in_site(
    store: Arc<dyn ObjectStore>,
    site_path: ObjectPath,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
) -> Result<Vec<NexradL2ArchiveInfo>> {
    let mut volumes = Vec::new();
    let mut list_stream = store.list(Some(&site_path));

    while let Some(result) = list_stream.next().await {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        let path = meta.location.to_string();
        let filename = path.split('/').next_back().unwrap_or("");

        // Skip MDM files
        if filename.ends_with("MDM") {
            continue;
        }

        // Parse filename
        let (instrument_name, vcp_time) = match parse_l2_filename(filename) {
            Ok(parsed) => parsed,
            Err(_) => continue,
        };

        // Filter by time range (only include volumes within bounds)
        if vcp_time < start_time || vcp_time >= end_time {
            continue;
        }

        volumes.push(NexradL2ArchiveInfo {
            uri: path.clone(),
            instrument_name,
            vcp_time,
            size: Some(meta.size),
            store: store.clone(),
            object_path: path,
        });
    }

    Ok(volumes)
}

/// Python wrapper for NexradL2ArchiveIterator
#[pyclass(from_py_object, module = "radrs._radrs")]
#[derive(Clone)]
pub struct NexradL2ArchiveIter {
    pub(crate) inner: NexradL2ArchiveIterator,
}

#[pymethods]
impl NexradL2ArchiveIter {
    #[new]
    #[pyo3(signature = (base_uri, start_time, end_time, storage_options=None, site_filter=None, max_concurrent_ls=10))]
    fn new(
        base_uri: String,
        start_time: &Bound<'_, PyDateTime>,
        end_time: &Bound<'_, PyDateTime>,
        storage_options: Option<HashMap<String, String>>,
        site_filter: Option<Vec<String>>,
        max_concurrent_ls: usize,
    ) -> PyResult<Self> {
        // Parse datetime objects
        let start_time_dt = parse_py_datetime(start_time)?;
        let end_time_dt = parse_py_datetime(end_time)?;

        if end_time_dt < start_time_dt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "end_time must be >= start_time",
            ));
        }

        let config = NexradL2ArchiveIterConfig {
            base_uri,
            start_time: start_time_dt,
            end_time: end_time_dt,
            storage_options,
            site_filter,
            max_concurrent_ls,
        };

        let inner = NexradL2ArchiveIterator::new(config)?;

        Ok(Self { inner })
    }

    /// Get next volume info
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<NexradL2ArchiveInfo>> {
        let runtime = crate::fetch::RUNTIME.handle();
        py.detach(|| runtime.block_on(self.inner.next()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
}

/// Python function to list L2 volumes for a time range
#[pyfunction]
#[pyo3(signature = (base_uri, start_time, end_time, storage_options=None, site_filter=None, max_concurrent_ls=10))]
pub fn list_nexrad_l2_archive_volumes_py(
    py: Python<'_>,
    base_uri: String,
    start_time: &Bound<'_, PyDateTime>,
    end_time: &Bound<'_, PyDateTime>,
    storage_options: Option<HashMap<String, String>>,
    site_filter: Option<Vec<String>>,
    max_concurrent_ls: usize,
) -> PyResult<Vec<NexradL2ArchiveInfo>> {
    let mut iter =
        NexradL2ArchiveIter::new(base_uri, start_time, end_time, storage_options, site_filter, max_concurrent_ls)?;

    let mut volumes = Vec::new();
    while let Some(vol) = iter.__next__(py)? {
        volumes.push(vol);
    }

    Ok(volumes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_l2_filename() {
        let (site, time) = parse_l2_filename("KTLX20240315_120000_V06").unwrap();
        assert_eq!(site, "KTLX");
        assert_eq!(time.format("%Y%m%d_%H%M%S").to_string(), "20240315_120000");
    }

    #[test]
    fn test_parse_l2_filename_invalid() {
        assert!(parse_l2_filename("invalid").is_err());
        assert!(parse_l2_filename("KTLX_20240315_120000").is_err());
    }
}
