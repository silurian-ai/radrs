//! Raystack batch accumulator with pre-allocation
//!
//! Pre-allocates flat arrays for P patterns, S sweeps, and R returns,
//! then fills incrementally from volume Scans (not dicts).

use crate::error::{RadrsError, Result};
use crate::fetch::RUNTIME;
use crate::iter::{VolumeMeta as PeekVolumeMeta, peek_volume_bytes, PEEK_SCAN_MAX};
use crate::metadata::extract_scan_meta;
use crate::qc;
use crate::raystack::QcOp;
use crate::raystack::parse::DEFAULT_FOLD_SIZE;
use crate::raystack::parse::QcArray;
use crate::raystack::parse::VolumeMeta;
use crate::raystack::parse::collect_metadata;
use crate::raystack::parse::ungzip_if_needed;
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{MomentData, MomentDataKind, MomentValue, Radial, Scan};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use std::collections::VecDeque;
use tokio::task::JoinHandle;

const PEEK_SCAN_LATLON: usize = PEEK_SCAN_MAX / 2;

// Asserts that a value is the same as another and returns the value
#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr) => {{
        let left_val = $left;
        let right_val = $right;
        if left_val != right_val {
            panic!(
                "assertion failed: `(left == right)`\n  left: `{:?}`,\n right: `{:?}`",
                left_val, right_val
            );
        }
        left_val
    }};
}

fn get_moment_datas(r: &Radial) -> [(MomentDataKind, Option<&MomentData>); 7] {
    return [
        (MomentDataKind::Reflectivity, r.reflectivity()),
        (MomentDataKind::Velocity, r.velocity()),
        (MomentDataKind::SpectrumWidth, r.spectrum_width()),
        (
            MomentDataKind::DifferentialReflectivity,
            r.differential_reflectivity(),
        ),
        (MomentDataKind::DifferentialPhase, r.differential_phase()),
        (
            MomentDataKind::CorrelationCoefficient,
            r.correlation_coefficient(),
        ),
        (MomentDataKind::ClutterFilterPower, r.clutter_filter_power()),
    ];
}

/// Pre-allocated raystack batch accumulator
///
/// Allocates flat arrays for max_patterns VCPs, max_sweeps sweeps,
/// and max_returns returns. Volumes are added incrementally until capacity is reached.
pub struct RaystackBatchData {
    // Capacity limits
    max_vcps: usize,
    max_sweeps: usize,
    max_returns: usize,
    fold_size: usize,
    truncate: bool,
    drop_empty_returns: bool,
    include_sweeps: bool,
    include_returns: bool,

    // Current fill indices
    n_vcps: usize,
    n_sweeps: usize,
    n_returns: usize,

    // Pattern/VCP metadata (filled incrementally, extended to max_patterns if truncate=false)
    source_fs_size: Vec<i64>,

    instrument_name: Vec<String>,
    instrument_type: Vec<String>,
    platform_type: Vec<String>,
    latitude: Vec<f32>,
    longitude: Vec<f32>,
    altitude: Vec<f32>,

    vcp_name: Vec<String>,
    vcp_number: Vec<u16>,
    vcp_time: Vec<i64>,
    vcp_duration: Vec<i64>,
    vcp_num_sweeps: Vec<u32>,

    // Sweep metadata (filled incrementally, extended to max_sweeps if truncate=false)
    sweep_vcp_time: Vec<i64>,
    sweep_number: Vec<u32>,
    sweep_time: Vec<i64>,
    sweep_duration: Vec<i64>,
    sweep_elevation_angle: Vec<f32>,
    sweep_elevation_number: Vec<u8>,
    sweep_range_start_m: Vec<f32>,
    sweep_range_step_m: Vec<f32>,
    sweep_max_range_m: Vec<f32>,
    sweep_max_gates: Vec<u32>,
    sweep_num_returns: Vec<u32>,

    // Return/radial coordinate data (grown incrementally, extended to max_returns if truncate=false)
    return_vcp_time: Vec<i64>,     // Parent VCP time for this return
    return_sweep_number: Vec<u32>, // Which sweep this return belongs to
    return_sweep_time: Vec<i64>,   // Parent sweep time for this return
    return_time: Vec<i64>,
    return_azimuth: Vec<f32>,
    return_elevation: Vec<f32>,
    return_base_range_m: Vec<f32>,
    return_range_step_m: Vec<f32>,

    // Moment data (grown incrementally, extended to max_returns * fold_size if truncate=false)
    // Using flat arrays for efficient numpy conversion
    dbzh: Vec<f32>,
    vradh: Vec<f32>,
    wradh: Vec<f32>,
    zdr: Vec<f32>,
    phidp: Vec<f32>,
    rhohv: Vec<f32>,
    ccorh: Vec<f32>,

    qc_outputs: Vec<(String, QcArray)>,

    is_finalized: bool,
}

impl RaystackBatchData {
    /// Create pre-allocated batch with specified capacity
    ///
    /// # Arguments
    /// * `max_patterns` - Maximum number of VCPs/patterns
    /// * `max_sweeps` - Maximum total number of sweeps across all patterns
    /// * `max_returns` - Maximum total number of returns/radials
    /// * `fold_size` - Range fold size (default 128)
    /// * `truncate` - Whether to truncate arrays to actual size on finalize (default true)
    pub fn new(
        max_vcps: usize,
        max_sweeps: usize,
        max_returns: usize,
        fold_size: usize,
        truncate: bool,
        drop_empty_returns: bool,
        include_sweeps: bool,
        include_returns: bool,
    ) -> Result<Self> {
        if max_vcps == 0
            || (include_sweeps && max_sweeps == 0)
            || (include_sweeps && include_returns && max_returns == 0)
        {
            return Err(RadrsError::InvalidInput(
                "All capacities must be greater than 0".into(),
            ));
        }

        let moment_capacity = max_returns * fold_size;

        Ok(Self {
            max_vcps,
            max_sweeps,
            max_returns,
            fold_size,
            truncate,
            drop_empty_returns,

            include_sweeps,
            include_returns: include_sweeps && include_returns,

            n_vcps: 0,
            n_sweeps: 0,
            n_returns: 0,

            // Pre-allocate pattern metadata (capacity only, will push during fill)
            source_fs_size: Vec::with_capacity(max_vcps),
            instrument_name: Vec::with_capacity(max_vcps),
            instrument_type: Vec::with_capacity(max_vcps),
            platform_type: Vec::with_capacity(max_vcps),
            latitude: Vec::with_capacity(max_vcps),
            longitude: Vec::with_capacity(max_vcps),
            altitude: Vec::with_capacity(max_vcps),
            vcp_name: Vec::with_capacity(max_vcps),
            vcp_number: Vec::with_capacity(max_vcps),
            vcp_time: Vec::with_capacity(max_vcps),
            vcp_duration: Vec::with_capacity(max_vcps),
            vcp_num_sweeps: Vec::with_capacity(max_vcps),

            // Pre-allocate sweep metadata (capacity only, will push during fill)
            sweep_vcp_time: Vec::with_capacity(max_sweeps),
            sweep_number: Vec::with_capacity(max_sweeps),
            sweep_time: Vec::with_capacity(max_sweeps),
            sweep_duration: Vec::with_capacity(max_sweeps),
            sweep_elevation_angle: Vec::with_capacity(max_sweeps),
            sweep_elevation_number: Vec::with_capacity(max_sweeps),
            sweep_range_start_m: Vec::with_capacity(max_sweeps),
            sweep_range_step_m: Vec::with_capacity(max_sweeps),
            sweep_max_range_m: Vec::with_capacity(max_sweeps),
            sweep_max_gates: Vec::with_capacity(max_sweeps),
            sweep_num_returns: Vec::with_capacity(max_sweeps),

            // Pre-allocate return coordinate arrays (capacity only, will grow as needed)
            return_vcp_time: Vec::with_capacity(max_returns),
            return_sweep_number: Vec::with_capacity(max_returns),
            return_sweep_time: Vec::with_capacity(max_returns),
            return_time: Vec::with_capacity(max_returns),
            return_azimuth: Vec::with_capacity(max_returns),
            return_elevation: Vec::with_capacity(max_returns),
            return_base_range_m: Vec::with_capacity(max_returns),
            return_range_step_m: Vec::with_capacity(max_returns),

            // Pre-allocate moment arrays (capacity only, will grow as needed)
            dbzh: Vec::with_capacity(moment_capacity),
            vradh: Vec::with_capacity(moment_capacity),
            wradh: Vec::with_capacity(moment_capacity),
            zdr: Vec::with_capacity(moment_capacity),
            phidp: Vec::with_capacity(moment_capacity),
            rhohv: Vec::with_capacity(moment_capacity),
            ccorh: Vec::with_capacity(moment_capacity),

            qc_outputs: Vec::new(),

            is_finalized: false,
        })
    }

    /// Add a volume from a cloud or local URL
    ///
    /// Supports S3, GCS, Azure, and local filesystem URLs.
    ///
    /// # Arguments
    /// * `url` - Full URL to the volume (e.g., "s3://bucket/path/to/volume", "/local/path/volume")
    /// * `storage_options` - Optional storage credentials and configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // S3 with anonymous access
    /// batch.add_volume_from_url(
    ///     "s3://noaa-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06",
    ///     Some(hashmap!{"anon" => "true"})
    /// ).await?;
    ///
    /// // GCS with service account
    /// batch.add_volume_from_url(
    ///     "gs://my-bucket/nexrad/KTLX20240315_120000_V06",
    ///     Some(hashmap!{"service_account_path" => "/path/to/key.json"})
    /// ).await?;
    ///
    /// // Local filesystem
    /// batch.add_volume_from_url("/data/nexrad/KTLX20240315_120000_V06", None).await?;
    /// ```
    pub async fn add_volume_from_url(
        &mut self,
        url: &str,
        storage_options: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        // Fetch the object from URL
        let bytes = crate::fetch::fetch_bytes_from_url(url, storage_options).await?;

        // Add to batch
        self.add_volume_bytes(&bytes, None)
    }

    /// Add volumes from L2 archive iterator with async prefetch support
    ///
    /// This is more efficient than manually calling `add_volume_bytes` in a loop
    /// because it uses async I/O with prefetch to overlap network fetches with parsing.
    ///
    /// # Arguments
    /// * `iter` - L2 archive iterator
    /// * `prefetch` - Number of volumes to fetch concurrently
    ///
    /// # Returns
    /// Number of volumes successfully added
    pub async fn add_volumes_from_l2_archive(
        &mut self,
        mut iter: crate::iter::NexradL2ArchiveIterator,
        prefetch: usize,
    ) -> Result<usize> {
        /// Helper to spawn a fetch task for the next volume
        async fn try_spawn_next(
            iter: &mut crate::iter::NexradL2ArchiveIterator,
            in_flight: &mut VecDeque<JoinHandle<Result<(Vec<u8>, u64)>>>,
            peek: bool,
        ) -> Result<bool> {
            match iter.next().await? {
                Some(volume_info) => {
                    let handle = tokio::spawn(async move {
                        volume_info
                            .fetch_with_size(if !peek { 0 } else { PEEK_SCAN_LATLON })
                            .await
                    });
                    in_flight.push_back(handle);
                    Ok(true)
                }
                None => Ok(false),
            }
        }

        let mut count = 0usize;
        let mut in_flight: VecDeque<JoinHandle<Result<(Vec<u8>, u64)>>> = VecDeque::new();

        // Fill the initial prefetch queue
        for _ in 0..prefetch {
            if !try_spawn_next(&mut iter, &mut in_flight, !self.include_sweeps).await? {
                break;
            }
        }

        // Process fetched volumes
        while let Some(handle) = in_flight.pop_front() {
            if !self.has_capacity() {
                break;
            }

            let result = handle
                .await
                .map_err(|e| RadrsError::Python(format!("Fetch task failed: {}", e)))?;

            match result {
                Ok((bytes, fs_size)) => {
                    // Try to add the volume
                    match self.add_volume_bytes(&bytes, Some(fs_size)) {
                        Ok(()) => {
                            count += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to add volume, skipping: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch volume: {}", e);
                }
            }

            // Refill prefetch queue if we still have capacity
            if self.has_capacity() && in_flight.len() < prefetch {
                let _ = try_spawn_next(&mut iter, &mut in_flight, !self.include_sweeps).await; // Ignore errors
            }
        }

        Ok(count)
    }

    /// Add a complete volume from raw bytes (single allocation path)
    pub fn add_volume_bytes(&mut self, data: &[u8], fs_size: Option<u64>) -> Result<()> {
        if self.is_finalized {
            return Err(RadrsError::InvalidInput("Batch is finalized".into()));
        }

        if !self.include_sweeps {
            let peek_meta = peek_volume_bytes(data, false).expect("Cannot peek in volume!");
            return self.add_peek(&peek_meta, fs_size.unwrap_or(data.len() as u64));
        }

        // Handle outer gzip
        let data = ungzip_if_needed(data)?;

        // Parse volume file
        let volume = VolumeFile::new(data.into_owned());
        let scan_meta = extract_scan_meta(&volume);
        let scan: Scan = volume.scan()?;
        let vol_meta = collect_metadata(&scan);

        return self.add_scan(&volume, &scan, &scan_meta, &vol_meta);
    }

    /// Add a VCP peek if we're just looking for basic metadata
    fn add_peek(&mut self, peek_meta: &PeekVolumeMeta, fs_size: u64) -> Result<()> {
        //
        // Capacity checks
        //

        if self.n_vcps + 1 > self.max_vcps {
            return Err(RadrsError::Capacity(format!(
                "VCP capacity exceeded: {} + {} > {}",
                self.n_vcps, 1, self.max_sweeps
            )));
        }

        //
        // VCP
        //

        self.source_fs_size.push(fs_size as i64);

        self.instrument_name.push(peek_meta.site.to_string());
        self.instrument_type.push("radar".to_string());
        self.platform_type.push("fixed".to_string());
        self.latitude.push(peek_meta.latitude.unwrap_or(f32::NAN));
        self.longitude.push(peek_meta.longitude.unwrap_or(f32::NAN));
        self.altitude.push(peek_meta.altitude.unwrap_or(f32::NAN));

        // Record VCP metadata
        self.vcp_name
            .push(format!("VCP-{}", peek_meta.vcp.unwrap_or(0)));
        self.vcp_number.push(peek_meta.vcp.unwrap_or(0));

        self.vcp_time
            .push(peek_meta.volume_datetime.timestamp_millis());
        self.vcp_duration.push(i64::MIN);
        self.vcp_num_sweeps.push(0);

        self.n_vcps += 1;

        return Ok(());
    }

    /// Add a scan directly (internal method)
    fn add_scan(
        &mut self,
        file: &VolumeFile,
        scan: &Scan,
        scan_meta: &crate::metadata::ScanMeta,
        vol_meta: &VolumeMeta,
    ) -> Result<()> {
        //
        // Capacity checks
        //

        if self.n_vcps + 1 > self.max_vcps {
            return Err(RadrsError::Capacity(format!(
                "VCP capacity exceeded: {} + {} > {}",
                self.n_vcps, 1, self.max_sweeps
            )));
        }

        if self.include_sweeps && self.n_sweeps + vol_meta.sweeps.len() > self.max_sweeps {
            return Err(RadrsError::Capacity(format!(
                "Sweep capacity exceeded: {} + {} > {}",
                self.n_sweeps,
                vol_meta.sweeps.len(),
                self.max_sweeps
            )));
        }

        if self.include_returns && self.n_returns + vol_meta.total_radials > self.max_returns {
            return Err(RadrsError::Capacity(format!(
                "Return capacity exceeded: {} + {} > {}",
                self.n_returns, vol_meta.total_radials, self.max_returns
            )));
        }

        let elevation_cuts = scan.coverage_pattern().elevation_cuts();

        //
        // VCP
        //

        self.source_fs_size.push(file.data().len() as i64);

        let inst_name = scan_meta.instrument_name.clone();
        if let Some(name) = inst_name {
            self.instrument_name.push(name);
            self.instrument_type.push("radar".to_string());
            self.platform_type.push("fixed".to_string());
        } else {
            self.instrument_name.push(String::new());
            self.instrument_type.push(String::new());
            self.platform_type.push(String::new());
        }
        let lat = scan_meta.latitude;
        let lon = scan_meta.longitude;
        let alt = scan_meta.altitude;
        self.latitude.push(lat.unwrap_or(f32::NAN));
        self.longitude.push(lon.unwrap_or(f32::NAN));
        self.altitude.push(alt.unwrap_or(f32::NAN));

        // Record VCP metadata
        self.vcp_name
            .push(format!("VCP-{}", scan.coverage_pattern_number()));
        self.vcp_number.push(scan.coverage_pattern_number());

        let vcp_time = vol_meta.min_time;
        self.vcp_time.push(vcp_time);
        self.vcp_duration.push(if vol_meta.max_time != i64::MIN {
            vol_meta.max_time - vol_meta.min_time
        } else {
            0
        });
        self.vcp_num_sweeps.push(vol_meta.sweeps.len() as u32);

        self.n_vcps += 1;

        if !self.include_sweeps {
            return Ok(());
        }

        //
        // SWEEPS
        //

        // Process each sweep

        let mut sweep_idx: i32 = -1;
        for sweep in scan.sweeps() {
            if sweep.radials().is_empty() {
                continue;
            }
            // Note we must count here due to empty sweeps
            sweep_idx += 1;

            let sweep_meta = vol_meta.sweeps[sweep_idx as usize];

            self.sweep_vcp_time.push(vcp_time);
            self.sweep_number.push(sweep_idx as u32);

            let sweep_time = sweep_meta.min_time;
            self.sweep_time.push(sweep_time);
            self.sweep_duration
                .push(sweep_meta.max_time - sweep_meta.min_time);

            self.sweep_elevation_number
                .push(sweep_meta.elevation_number);
            // Nominal sweep angles are stored in VCP-level metadata "elevation cuts"
            self.sweep_elevation_angle.push(
                if (sweep_meta.elevation_number as usize) <= elevation_cuts.len() {
                    elevation_cuts[(sweep_meta.elevation_number - 1) as usize]
                        .elevation_angle_degrees() as f32
                } else {
                    sweep_meta.elevation_angle
                },
            );
            self.sweep_max_gates.push(sweep_meta.max_gates as u32);
            self.sweep_range_start_m
                .push((sweep_meta.range_first_km * 1000.0) as f32);
            self.sweep_range_step_m
                .push((sweep_meta.gate_interval_km * 1000.0) as f32);
            // Computation is to sum the distance of the first + (all but the first) gates
            self.sweep_max_range_m.push(
                ((sweep_meta.range_first_km
                    + sweep_meta.gate_interval_km * ((sweep_meta.max_gates - 1) as f64))
                    * 1000.0) as f32,
            );

            if !self.include_returns {
                self.sweep_num_returns.push(0);
                self.n_sweeps += 1;
                continue;
            }

            let start_sweep_return_idx = self.n_returns;

            //
            // Returns
            //

            // Values are computed so stored once for multiple folds
            let mut moment_values: [Option<Vec<MomentValue>>; 7] = [const { None }; 7];

            for radial in sweep.radials() {
                let mut rad_max_gates = 0u16;
                let mut rad_first_gate_km = 0.0f32;
                let mut rad_gate_step_km = 0.0f32;

                // First pass: compute radial metadata and cache moment values
                for (m_idx, (_, maybe_m)) in get_moment_datas(radial).iter().enumerate() {
                    if let Some(m) = maybe_m {
                        rad_max_gates = rad_max_gates.max(m.gate_count());
                        rad_first_gate_km = if rad_first_gate_km == 0.0 {
                            m.first_gate_range_km() as f32
                        } else {
                            ensure_eq!(rad_first_gate_km, m.first_gate_range_km() as f32)
                        };
                        rad_gate_step_km = if rad_gate_step_km == 0.0 {
                            m.gate_interval_km() as f32
                        } else {
                            ensure_eq!(rad_gate_step_km, m.gate_interval_km() as f32)
                        };
                        moment_values[m_idx] = Some(m.values());
                    } else {
                        moment_values[m_idx] = None;
                    }
                }

                // // ceil int division
                let n_folds = ((rad_max_gates as usize) + self.fold_size - 1) / self.fold_size;
                for f in 0..n_folds {
                    let base_gate = f * self.fold_size;
                    let start_m_val_idx = base_gate;
                    let end_m_val_idx = base_gate + self.fold_size;
                    let start_m_out_idx = self.n_returns * self.fold_size;
                    let end_m_out_idx = start_m_out_idx + self.fold_size;

                    let mut n_finite_values = 0usize;

                    // Second pass: fill moment data into output vectors
                    for (m_idx, (kind, maybe_m)) in get_moment_datas(radial).iter().enumerate() {
                        let m_out = self.get_moment_vectors(*kind);
                        assert_eq!(
                            start_m_out_idx,
                            m_out.len(),
                            "Bad data length for type={} at nreturns={}",
                            *kind as u8,
                            self.n_returns
                        );
                        m_out.resize(end_m_out_idx, f32::NAN);

                        if let Some(_) = maybe_m {
                            let m_vals: &Vec<MomentValue> = moment_values[m_idx].as_ref().unwrap();

                            let trunc_m_val_idx = end_m_val_idx.min(m_vals.len());
                            if trunc_m_val_idx <= start_m_val_idx {
                                continue;
                            };

                            for i in 0..(trunc_m_val_idx - start_m_val_idx) {
                                m_out[start_m_out_idx + i] = match m_vals[start_m_val_idx + i] {
                                    MomentValue::Value(x) => x,
                                    _ => f32::NAN,
                                };

                                if m_out[start_m_out_idx + i].is_finite() {
                                    n_finite_values += 1;
                                }
                            }
                        }
                    }

                    if self.drop_empty_returns && n_finite_values == 0 {
                        // Roll back all moment vectors if this return has no finite values
                        for (kind, _) in get_moment_datas(radial) {
                            let m_out = self.get_moment_vectors(kind);
                            m_out.resize(start_m_out_idx, f32::NAN);
                        }
                        // Don't write anything for empty data
                        continue;
                    }

                    self.return_vcp_time.push(vcp_time);
                    self.return_sweep_number.push(sweep_idx as u32);
                    self.return_sweep_time.push(sweep_time);

                    self.return_time.push(radial.collection_timestamp());
                    self.return_azimuth.push(radial.azimuth_angle_degrees());
                    self.return_elevation.push(radial.elevation_angle_degrees());

                    self.return_base_range_m.push(
                        (rad_first_gate_km + (rad_gate_step_km * (base_gate as f32))) * 1000.0,
                    );
                    self.return_range_step_m.push(rad_gate_step_km * 1000.0);

                    self.n_returns += 1;
                }
            }

            self.sweep_num_returns
                .push((self.n_returns - start_sweep_return_idx) as u32);
            self.n_sweeps += 1;
        }

        Ok(())
    }

    fn get_moment_vectors(&mut self, kind: MomentDataKind) -> &mut Vec<f32> {
        match kind {
            MomentDataKind::Reflectivity => &mut self.dbzh,
            MomentDataKind::Velocity => &mut self.vradh,
            MomentDataKind::SpectrumWidth => &mut self.wradh,
            MomentDataKind::DifferentialReflectivity => &mut self.zdr,
            MomentDataKind::DifferentialPhase => &mut self.phidp,
            MomentDataKind::CorrelationCoefficient => &mut self.rhohv,
            MomentDataKind::ClutterFilterPower => &mut self.ccorh,
            _ => unreachable!("Unexpected moment data kind: {}", kind as u8),
        }
    }

    /// Get current fill progress
    pub fn progress(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.n_vcps,
            self.max_vcps,
            self.n_sweeps,
            self.max_sweeps,
            self.n_returns,
            self.max_returns,
        )
    }

    /// Check if batch has remaining capacity
    pub fn has_capacity(&self) -> bool {
        !self.is_finalized
            && self.n_vcps < self.max_vcps
            && (!self.include_sweeps || self.n_sweeps < self.max_sweeps)
            && (!self.include_returns || self.n_returns < self.max_returns)
    }

    /// Finalize batch (optionally extend to full capacity)
    ///
    /// If truncate=true, arrays remain at actual filled size (no-op).
    /// If truncate=false, extends all arrays to max capacity with fill values (NaN/0).
    /// This allows for fixed-size output tensors for ML pipelines.
    pub fn finalize(&mut self) {
        if self.is_finalized {
            return;
        }

        if !self.truncate {
            // Extend VCP/pattern metadata arrays to max capacity with fill values
            self.source_fs_size.resize(self.max_vcps, 0);
            self.instrument_name.resize(self.max_vcps, String::new());
            self.instrument_type.resize(self.max_vcps, String::new());
            self.platform_type.resize(self.max_vcps, String::new());
            self.latitude.resize(self.max_vcps, f32::NAN);
            self.longitude.resize(self.max_vcps, f32::NAN);
            self.altitude.resize(self.max_vcps, f32::NAN);
            self.vcp_name.resize(self.max_vcps, String::new());
            self.vcp_number.resize(self.max_vcps, 0);
            self.vcp_time.resize(self.max_vcps, i64::MIN); // NaT for numpy datetime64
            self.vcp_duration.resize(self.max_vcps, i64::MIN); // NaT for numpy timedelta64
            self.vcp_num_sweeps.resize(self.max_vcps, 0);

            if self.include_sweeps {
                // Extend sweep metadata arrays to max capacity with fill values
                self.sweep_vcp_time.resize(self.max_sweeps, i64::MIN);
                self.sweep_number.resize(self.max_sweeps, 0);
                self.sweep_time.resize(self.max_sweeps, i64::MIN); // NaT for numpy datetime64
                self.sweep_duration.resize(self.max_sweeps, i64::MIN);
                self.sweep_elevation_angle.resize(self.max_sweeps, f32::NAN);
                self.sweep_elevation_number.resize(self.max_sweeps, 0);
                self.sweep_max_gates.resize(self.max_sweeps, 0);
                self.sweep_range_start_m.resize(self.max_sweeps, f32::NAN);
                self.sweep_range_step_m.resize(self.max_sweeps, f32::NAN);
                self.sweep_max_range_m.resize(self.max_sweeps, f32::NAN);
                self.sweep_num_returns.resize(self.max_sweeps, 0);
            }

            if self.include_returns {
                // Extend return coordinate arrays to max capacity with fill values
                self.return_vcp_time.resize(self.max_returns, i64::MIN); // NaT for numpy datetime64
                self.return_sweep_number.resize(self.max_returns, 0);
                self.return_sweep_time.resize(self.max_returns, i64::MIN); // NaT for numpy datetime64
                self.return_time.resize(self.max_returns, i64::MIN); // NaT for numpy datetime64
                self.return_azimuth.resize(self.max_returns, f32::NAN);
                self.return_elevation.resize(self.max_returns, f32::NAN);
                self.return_base_range_m.resize(self.max_returns, f32::NAN);
                self.return_range_step_m.resize(self.max_returns, f32::NAN);

                // Extend moment arrays to max capacity with fill values
                let max_moment_len = self.max_returns * self.fold_size;
                self.dbzh.resize(max_moment_len, f32::NAN);
                self.vradh.resize(max_moment_len, f32::NAN);
                self.wradh.resize(max_moment_len, f32::NAN);
                self.zdr.resize(max_moment_len, f32::NAN);
                self.phidp.resize(max_moment_len, f32::NAN);
                self.rhohv.resize(max_moment_len, f32::NAN);
                self.ccorh.resize(max_moment_len, f32::NAN);

                for (_name, arr) in &mut self.qc_outputs {
                    match arr {
                        QcArray::Mask(mask) => {
                            mask.resize(max_moment_len, 0);
                        }
                        QcArray::Float(floats) => {
                            floats.resize(max_moment_len, f32::NAN);
                        }
                    }
                }
            }
        }
        // If truncate=true, all arrays remain at actual filled size (no action needed)

        self.is_finalized = true;
    }

    fn add_qc_outputs(&mut self, qc_ops: &[QcOp]) {
        for op in qc_ops {
            match op {
                QcOp::RhohvThreshold { threshold, vname } => {
                    let mask: Vec<i8> = qc::rhohv_threshold(&self.rhohv, *threshold);
                    self.qc_outputs.push((vname.clone(), QcArray::Mask(mask)));
                }
                QcOp::SunSpike {
                    dbzh_threshold,
                    fill_threshold,
                    corr_threshold,
                    vname,
                } => {
                    let mask = qc::sun_spike(
                        &self.dbzh,
                        self.n_returns,
                        self.fold_size,
                        *dbzh_threshold,
                        *fill_threshold,
                        *corr_threshold,
                    );
                    self.qc_outputs.push((vname.clone(), QcArray::Mask(mask)));
                }
                QcOp::VradhWindingNumber {
                    nyquist,
                    wind_size,
                    velocity_texture_threshold,
                    reflectivity_threshold,
                    interval_splits,
                    skip_between_rays,
                    skip_along_ray,
                    centered,
                    rays_wrap_around,
                    fill_value,
                    fill_tolerance,
                    vname,
                } => {
                    let mut out = vec![f32::NAN; self.n_returns * self.fold_size];
                    let mut sweep_start_index = 0;
                    for sweep_idx in 0..self.sweep_time.len() {
                        let start = sweep_start_index;
                        let n_radials = self.sweep_num_returns[sweep_idx] as usize;
                        sweep_start_index += n_radials;
                        let fold_size = self.fold_size;
                        let slice_len = n_radials * fold_size;
                        let offset = start * fold_size;
                        let vradh = &self.vradh[offset..offset + slice_len];
                        let dbzh = &self.dbzh[offset..offset + slice_len];
                        let params = qc::VradhWindingParams {
                            nyquist: *nyquist,
                            wind_size: *wind_size,
                            velocity_texture_threshold: *velocity_texture_threshold,
                            reflectivity_threshold: *reflectivity_threshold,
                            interval_splits: *interval_splits,
                            skip_between_rays: *skip_between_rays,
                            skip_along_ray: *skip_along_ray,
                            centered: *centered,
                            rays_wrap_around: *rays_wrap_around,
                            fill_value: *fill_value,
                            fill_tolerance: *fill_tolerance,
                        };
                        let sweep_out = qc::vradh_winding_number(
                            vradh,
                            Some(dbzh),
                            n_radials,
                            fold_size,
                            params,
                        );
                        out[offset..offset + slice_len].copy_from_slice(&sweep_out);
                    }
                    self.qc_outputs.push((vname.clone(), QcArray::Float(out)));
                }
            }
        }
    }

    /// Convert to Python dict (for compatibility with existing tools)
    pub fn to_python_dict(mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        use numpy::IntoPyArray;
        self.finalize();

        let dict = PyDict::new(py);

        let vcps_dict = PyDict::new(py);

        vcps_dict.set_item("source_fs_size", self.source_fs_size.into_pyarray(py))?;

        vcps_dict.set_item("instrument_name", self.instrument_name)?;
        vcps_dict.set_item("instrument_type", self.instrument_type)?;
        vcps_dict.set_item("platform_type", self.platform_type)?;
        vcps_dict.set_item("latitude", self.latitude.into_pyarray(py))?;
        vcps_dict.set_item("longitude", self.longitude.into_pyarray(py))?;
        vcps_dict.set_item("altitude", self.altitude.into_pyarray(py))?;

        vcps_dict.set_item("vcp_name", self.vcp_name)?;
        vcps_dict.set_item("vcp_number", self.vcp_number.into_pyarray(py))?;
        vcps_dict.set_item("vcp_time", self.vcp_time.into_pyarray(py))?;
        vcps_dict.set_item("vcp_duration", self.vcp_duration.into_pyarray(py))?;
        vcps_dict.set_item("num_sweeps", self.vcp_num_sweeps.into_pyarray(py))?;

        dict.set_item("vcps", vcps_dict)?;

        if self.include_sweeps {
            let sweeps_dict = PyDict::new(py);

            sweeps_dict.set_item("vcp_time", self.sweep_vcp_time.into_pyarray(py))?;
            sweeps_dict.set_item("sweep_number", self.sweep_number.into_pyarray(py))?;
            sweeps_dict.set_item("sweep_time", self.sweep_time.into_pyarray(py))?;
            sweeps_dict.set_item("sweep_duration", self.sweep_duration.into_pyarray(py))?;
            sweeps_dict.set_item(
                "elevation_angle",
                self.sweep_elevation_angle.into_pyarray(py),
            )?;
            sweeps_dict.set_item(
                "elevation_number",
                self.sweep_elevation_number.into_pyarray(py),
            )?;
            sweeps_dict.set_item("range_start", self.sweep_range_start_m.into_pyarray(py))?;
            sweeps_dict.set_item("range_step", self.sweep_range_step_m.into_pyarray(py))?;
            sweeps_dict.set_item("max_range", self.sweep_max_range_m.into_pyarray(py))?;
            sweeps_dict.set_item("max_gates", self.sweep_max_gates.into_pyarray(py))?;
            sweeps_dict.set_item("num_returns", self.sweep_num_returns.into_pyarray(py))?;

            dict.set_item("sweeps", sweeps_dict)?;
        }

        if self.include_returns {
            let returns_dict = PyDict::new(py);

            // Convert coordinate arrays to numpy
            returns_dict.set_item("vcp_time", self.return_vcp_time.into_pyarray(py))?;
            returns_dict.set_item("sweep_number", self.return_sweep_number.into_pyarray(py))?;
            returns_dict.set_item("sweep_time", self.return_sweep_time.into_pyarray(py))?;
            returns_dict.set_item("return_time", self.return_time.into_pyarray(py))?;
            returns_dict.set_item("azimuth", self.return_azimuth.into_pyarray(py))?;
            returns_dict.set_item("elevation", self.return_elevation.into_pyarray(py))?;
            returns_dict.set_item("base_range", self.return_base_range_m.into_pyarray(py))?;
            returns_dict.set_item("range_step", self.return_range_step_m.into_pyarray(py))?;

            returns_dict.set_item(
                "range",
                (0..self.fold_size as u32)
                    .collect::<Vec<u32>>()
                    .into_pyarray(py),
            )?;

            returns_dict.set_item("DBZH", self.dbzh.into_pyarray(py))?;
            returns_dict.set_item("VRADH", self.vradh.into_pyarray(py))?;
            returns_dict.set_item("WRADH", self.wradh.into_pyarray(py))?;
            returns_dict.set_item("ZDR", self.zdr.into_pyarray(py))?;
            returns_dict.set_item("PHIDP", self.phidp.into_pyarray(py))?;
            returns_dict.set_item("RHOHV", self.rhohv.into_pyarray(py))?;
            returns_dict.set_item("CCORH", self.ccorh.into_pyarray(py))?;

            for (name, arr) in self.qc_outputs {
                match arr {
                    QcArray::Mask(mask) => {
                        returns_dict.set_item(format!("qc.{}", name), mask.into_pyarray(py))?;
                    }
                    QcArray::Float(floats) => {
                        returns_dict.set_item(format!("qc.{}", name), floats.into_pyarray(py))?;
                    }
                }
            }

            dict.set_item("returns", returns_dict)?;
        }

        Ok(dict.into())
    }
}

/// Python wrapper for RaystackBatchData
#[pyclass(name = "BatchedRaystack")]
pub struct BatchedRaystackPy {
    inner: Option<RaystackBatchData>,
}

#[pymethods]
impl BatchedRaystackPy {
    #[new]
    #[pyo3(signature = (max_vcps, max_sweeps, max_returns, fold_size=DEFAULT_FOLD_SIZE, truncate=true, drop_empty_returns=false, include_sweeps=true, include_returns=true))]
    fn new(
        max_vcps: usize,
        max_sweeps: usize,
        max_returns: usize,
        fold_size: usize,
        truncate: bool,
        drop_empty_returns: bool,
        include_sweeps: bool,
        include_returns: bool,
    ) -> PyResult<Self> {
        let inner = RaystackBatchData::new(
            max_vcps,
            max_sweeps,
            max_returns,
            fold_size,
            truncate,
            drop_empty_returns,
            include_sweeps,
            include_returns,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: Some(inner) })
    }

    fn add_volume(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = data.as_bytes();
        self.inner
            .as_mut()
            .unwrap()
            .add_volume_bytes(bytes, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (url, storage_options=None))]
    fn add_volume_from_url(
        &mut self,
        py: Python<'_>,
        url: String,
        storage_options: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().unwrap();
        // Run the async method in blocking mode
        py.detach(|| {
            RUNTIME.block_on(async { inner.add_volume_from_url(&url, storage_options).await })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (l2_iter, prefetch=2))]
    fn add_volumes_from_l2(
        &mut self,
        py: Python<'_>,
        l2_iter: &Bound<'_, PyAny>,
        prefetch: usize,
    ) -> PyResult<usize> {
        // Extract the NexradL2ArchiveIter and get its inner iterator
        let iter_wrapper: crate::iter::NexradL2ArchiveIter = l2_iter.extract()?;
        let inner = self.inner.as_mut().unwrap();

        // Run the async method in blocking mode
        py.detach(|| {
            RUNTIME.block_on(async {
                inner
                    .add_volumes_from_l2_archive(iter_wrapper.inner, prefetch)
                    .await
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn progress(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (n_pat, max_pat, n_sw, max_sw, n_ret, max_ret) =
            self.inner.as_ref().unwrap().progress();
        let dict = PyDict::new(py);
        dict.set_item("patterns_filled", n_pat)?;
        dict.set_item("patterns_capacity", max_pat)?;
        dict.set_item("sweeps_filled", n_sw)?;
        dict.set_item("sweeps_capacity", max_sw)?;
        dict.set_item("returns_filled", n_ret)?;
        dict.set_item("returns_capacity", max_ret)?;
        dict.set_item("fill_fraction", n_ret as f32 / max_ret as f32)?;
        Ok(dict.into())
    }

    fn has_capacity(&self) -> bool {
        self.inner.as_ref().unwrap().has_capacity()
    }

    fn add_qc_outputs(&mut self, py: Python<'_>, qc_ops: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let ops = crate::raystack::parse::parse_qc_ops(py, qc_ops.as_ref())?;
        self.inner.as_mut().unwrap().add_qc_outputs(&ops);
        Ok(())
    }

    fn finalize(&mut self) {
        self.inner.as_mut().unwrap().finalize()
    }

    fn finalize_to_dict(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.inner.take().unwrap().to_python_dict(py)
    }

    fn __repr__(&self) -> String {
        let (n_pat, max_pat, n_sw, max_sw, n_ret, max_ret) =
            self.inner.as_ref().unwrap().progress();
        format!(
            "BatchedRaystack(patterns={}/{}, sweeps={}/{}, returns={}/{}, fill={:.1}%)",
            n_pat,
            max_pat,
            n_sw,
            max_sw,
            n_ret,
            max_ret,
            (n_ret as f32 / max_ret as f32) * 100.0
        )
    }
}
