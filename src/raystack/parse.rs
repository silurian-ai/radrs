//! Optimized direct parsing to raystack format
//!
//! Uses two-pass parsing with preallocation for minimal allocations.
//! Supports parallel BZ2 decompression via rayon.

use crate::constants::{INSTRUMENT_TYPE, MOMENT_NAMES, PLATFORM_TYPE};
use crate::error::{RadrsError, Result};
use crate::fetch::{RUNTIME, fetch_s3_url};
use crate::metadata::{ScanMeta, extract_scan_meta};
use crate::metadata_build::{build_root_attrs, set_sweep_mode_vars};
use crate::qc;
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{MomentValue, Scan};
use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};
use pyo3_async_runtimes::tokio::future_into_py;
use std::borrow::Cow;
use std::fs;
use std::io::Read;
use std::time::Instant;

/// Default fold size for raystack format
pub const DEFAULT_FOLD_SIZE: usize = 128;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Moment indices for fixed-size arrays
const MOMENT_DBZH: usize = 0;
const MOMENT_VRADH: usize = 1;
const MOMENT_WRADH: usize = 2;
const MOMENT_ZDR: usize = 3;
const MOMENT_PHIDP: usize = 4;
const MOMENT_RHOHV: usize = 5;
const MOMENT_KDP: usize = 6;

/// Decompress outer gzip if present. Uses Cow to avoid allocation when not gzipped.
fn ungzip_if_needed(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    if data.starts_with(&GZIP_MAGIC) {
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(Cow::Owned(decompressed))
    } else {
        Ok(Cow::Borrowed(data))
    }
}

/// Sweep metadata collected in first pass
#[derive(Clone, Copy, Default)]
struct SweepMeta {
    elevation_number: u8,
    elevation_angle: f32,
    n_radials: usize,
    max_gates: usize,      // Max gate count across all moments in this sweep
    range_first_km: f64,   // First gate range for sweep grid (km)
    gate_interval_km: f64, // Gate interval for sweep grid (km)
}

/// Volume metadata from first pass
struct VolumeMeta {
    pattern_number: u16,
    sweeps: Vec<SweepMeta>,
    total_radials: usize,
}

/// Preallocated raystack data structure
pub struct RaystackData {
    pub pattern_number: u16,
    pub instrument_name: Option<String>,
    pub latitude: Option<f32>,
    pub longitude: Option<f32>,
    pub altitude: Option<f32>,
    pub sweeps: Vec<SweepInfo>,
    pub n_radials: usize,
    pub fold_size: usize,

    // Flat coordinate arrays (length = n_radials)
    pub azimuth: Vec<f32>,
    pub elevation: Vec<f32>,
    pub time: Vec<i64>,
    pub sweep_idx: Vec<u32>,

    // Flat moment arrays (length = n_radials * fold_size)
    // Using flat arrays avoids Vec<Vec<>> overhead
    pub dbzh: Vec<f32>,
    pub vradh: Vec<f32>,
    pub wradh: Vec<f32>,
    pub zdr: Vec<f32>,
    pub phidp: Vec<f32>,
    pub rhohv: Vec<f32>,
    pub kdp: Vec<f32>,
}

struct ActivityData {
    moments: Vec<&'static str>,
    ray_valid_count: Vec<u32>,
    ray_valid_fraction: Vec<f32>,
    sweep_valid_count: Vec<u32>,
    sweep_valid_fraction: Vec<f32>,
    volume_valid_count: Vec<u32>,
    volume_valid_fraction: Vec<f32>,
    n_returns: usize,
    n_sweeps: usize,
}

fn compute_activity(raystack: &RaystackData) -> ActivityData {
    let moment_data: [&[f32]; 7] = [
        &raystack.dbzh,
        &raystack.vradh,
        &raystack.wradh,
        &raystack.zdr,
        &raystack.phidp,
        &raystack.rhohv,
        &raystack.kdp,
    ];
    let n_moments = MOMENT_NAMES.len();
    let n_returns = raystack.n_radials;
    let n_sweeps = raystack.sweeps.len();
    let fold_size = raystack.fold_size;

    let mut ray_valid_count = vec![0u32; n_moments * n_returns];
    let mut ray_valid_fraction = vec![f32::NAN; n_moments * n_returns];

    if n_returns > 0 && fold_size > 0 {
        for (m_idx, data) in moment_data.iter().enumerate() {
            for (r_idx, chunk) in data.chunks_exact(fold_size).enumerate() {
                let count = chunk.iter().filter(|v| v.is_finite()).count() as u32;
                let idx = m_idx * n_returns + r_idx;
                if idx < ray_valid_count.len() {
                    ray_valid_count[idx] = count;
                    ray_valid_fraction[idx] = count as f32 / fold_size as f32;
                }
            }
        }
    }

    let mut sweep_valid_count = vec![0u32; n_moments * n_sweeps];
    let mut sweep_valid_fraction = vec![f32::NAN; n_moments * n_sweeps];

    if fold_size > 0 && n_sweeps > 0 {
        for (s_idx, sweep) in raystack.sweeps.iter().enumerate() {
            let start = sweep.start_index.min(n_returns);
            let end = (start + sweep.n_radials).min(n_returns);
            let sweep_radials = end.saturating_sub(start);
            let denom = sweep_radials * fold_size;
            for m_idx in 0..n_moments {
                let base = m_idx * n_returns;
                let count: u32 = ray_valid_count[base + start..base + end]
                    .iter()
                    .copied()
                    .sum();
                let idx = m_idx * n_sweeps + s_idx;
                sweep_valid_count[idx] = count;
                sweep_valid_fraction[idx] = if denom > 0 {
                    count as f32 / denom as f32
                } else {
                    f32::NAN
                };
            }
        }
    }

    let mut volume_valid_count = vec![0u32; n_moments];
    let mut volume_valid_fraction = vec![f32::NAN; n_moments];
    let denom = n_returns * fold_size;
    for m_idx in 0..n_moments {
        let base = m_idx * n_returns;
        let total: u32 = ray_valid_count[base..base + n_returns]
            .iter()
            .copied()
            .sum();
        volume_valid_count[m_idx] = total;
        volume_valid_fraction[m_idx] = if denom > 0 {
            total as f32 / denom as f32
        } else {
            f32::NAN
        };
    }

    ActivityData {
        moments: MOMENT_NAMES.to_vec(),
        ray_valid_count,
        ray_valid_fraction,
        sweep_valid_count,
        sweep_valid_fraction,
        volume_valid_count,
        volume_valid_fraction,
        n_returns,
        n_sweeps,
    }
}

/// Sweep-level metadata (public)
#[derive(Clone)]
pub struct SweepInfo {
    pub sweep_number: u32,
    pub elevation_number: u8,
    pub elevation_angle: f32,
    pub n_radials: usize,
    pub start_index: usize,
    pub max_gates: usize,
    pub range_first_km: f64,
    pub gate_interval_km: f64,
}

#[derive(Clone)]
pub enum QcOp {
    RhohvThreshold {
        threshold: f32,
        vname: String,
    },
    SunSpike {
        dbzh_threshold: f32,
        fill_threshold: f32,
        corr_threshold: f32,
        vname: String,
    },
    VradhWindingNumber {
        nyquist: Option<f32>,
        wind_size: usize,
        velocity_texture_threshold: f32,
        reflectivity_threshold: f32,
        interval_splits: usize,
        skip_between_rays: usize,
        skip_along_ray: usize,
        centered: bool,
        rays_wrap_around: bool,
        fill_value: Option<f32>,
        fill_tolerance: f32,
        vname: String,
    },
}

enum QcArray {
    Mask(Vec<i8>),
    Float(Vec<f32>),
}

fn sweep_range_metrics(sweep: &SweepInfo, fold_size: usize) -> (f32, f32, f32) {
    let range_start_m = (sweep.range_first_km * 1000.0) as f32;
    let gate_interval_m = (sweep.gate_interval_km * 1000.0) as f32;

    if fold_size == 0 || sweep.max_gates == 0 || gate_interval_m <= 0.0 {
        return (range_start_m, f32::NAN, f32::NAN);
    }

    let range_step_m = if sweep.max_gates <= fold_size {
        gate_interval_m
    } else {
        gate_interval_m * (sweep.max_gates as f32 / fold_size as f32)
    };

    let max_range_m = range_start_m + gate_interval_m * ((sweep.max_gates - 1) as f32);

    (range_start_m, range_step_m, max_range_m)
}

impl RaystackData {
    /// Allocate with known sizes
    fn with_capacity(meta: &VolumeMeta, fold_size: usize, meta_info: &ScanMeta) -> Self {
        let n = meta.total_radials;
        let moment_len = n * fold_size;

        let mut sweeps = Vec::with_capacity(meta.sweeps.len());
        let mut start_index = 0;
        for (idx, sm) in meta.sweeps.iter().enumerate() {
            sweeps.push(SweepInfo {
                sweep_number: idx as u32,
                elevation_number: sm.elevation_number,
                elevation_angle: sm.elevation_angle,
                n_radials: sm.n_radials,
                start_index,
                max_gates: sm.max_gates,
                range_first_km: sm.range_first_km,
                gate_interval_km: sm.gate_interval_km,
            });
            start_index += sm.n_radials;
        }

        Self {
            pattern_number: meta.pattern_number,
            instrument_name: meta_info.instrument_name.clone(),
            latitude: meta_info.latitude,
            longitude: meta_info.longitude,
            altitude: meta_info.altitude,
            sweeps,
            n_radials: n,
            fold_size,
            azimuth: vec![0.0; n],
            elevation: vec![0.0; n],
            time: vec![0; n],
            sweep_idx: vec![0; n],
            dbzh: vec![f32::NAN; moment_len],
            vradh: vec![f32::NAN; moment_len],
            wradh: vec![f32::NAN; moment_len],
            zdr: vec![f32::NAN; moment_len],
            phidp: vec![f32::NAN; moment_len],
            rhohv: vec![f32::NAN; moment_len],
            kdp: vec![f32::NAN; moment_len],
        }
    }

    /// Get mutable slice for a moment at given radial index
    #[inline]
    fn moment_slice_mut(&mut self, moment_idx: usize, radial_idx: usize) -> &mut [f32] {
        let start = radial_idx * self.fold_size;
        let end = start + self.fold_size;
        match moment_idx {
            MOMENT_DBZH => &mut self.dbzh[start..end],
            MOMENT_VRADH => &mut self.vradh[start..end],
            MOMENT_WRADH => &mut self.wradh[start..end],
            MOMENT_ZDR => &mut self.zdr[start..end],
            MOMENT_PHIDP => &mut self.phidp[start..end],
            MOMENT_RHOHV => &mut self.rhohv[start..end],
            MOMENT_KDP => &mut self.kdp[start..end],
            _ => unreachable!(),
        }
    }
}

pub(crate) fn parse_qc_ops(py: Python<'_>, qc: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<QcOp>> {
    let Some(qc_any) = qc else {
        return Ok(Vec::new());
    };

    if qc_any.is_none() {
        return Ok(Vec::new());
    }

    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = qc_any.cast::<PyList>() {
        list.iter().map(|item| item.to_owned()).collect()
    } else {
        vec![qc_any.to_owned()]
    };

    let mut ops = Vec::new();
    for item in items {
        if let Ok(name) = item.extract::<String>() {
            ops.push(qc_op_from_name(&name, None, py)?);
            continue;
        }

        if let Ok(dict) = item.cast::<PyDict>() {
            let name_any = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("QC spec missing 'name'"))?;
            let name: String = name_any.extract()?;
            ops.push(qc_op_from_name(&name, Some(&dict), py)?);
            continue;
        }

        return Err(pyo3::exceptions::PyTypeError::new_err(
            "qc must be a QC spec or list of specs",
        ));
    }

    Ok(ops)
}

fn dict_f32(dict: &Bound<'_, PyDict>, key: &str, default: f32) -> PyResult<f32> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_string(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default.to_string());
        }
        return value.extract();
    }
    Ok(default.to_string())
}

fn dict_usize(dict: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_opt_f32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f32>> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(None);
        }
        let v: f32 = value.extract()?;
        return Ok(Some(v));
    }
    Ok(None)
}

fn qc_op_from_name(
    name: &str,
    dict: Option<&Bound<'_, PyDict>>,
    _py: Python<'_>,
) -> PyResult<QcOp> {
    match name {
        "rhohv_threshold" => {
            let threshold = if let Some(dict) = dict {
                dict_f32(dict, "threshold", 0.8)?
            } else {
                0.8
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "rhohv_threshold_mask")?
            } else {
                "rhohv_threshold_mask".to_string()
            };
            Ok(QcOp::RhohvThreshold { threshold, vname })
        }
        "sun_spike" => {
            let dbzh_threshold = if let Some(dict) = dict {
                dict_f32(dict, "dbzh_threshold", 0.0)?
            } else {
                0.0
            };
            let fill_threshold = if let Some(dict) = dict {
                dict_f32(dict, "fill_threshold", 0.9)?
            } else {
                0.9
            };
            let corr_threshold = if let Some(dict) = dict {
                dict_f32(dict, "corr_threshold", 0.8)?
            } else {
                0.8
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "sun_spike_mask")?
            } else {
                "sun_spike_mask".to_string()
            };
            Ok(QcOp::SunSpike {
                dbzh_threshold,
                fill_threshold,
                corr_threshold,
                vname,
            })
        }
        "vradh_winding_number" => {
            let nyquist = if let Some(dict) = dict {
                dict_opt_f32(dict, "nyquist")?
            } else {
                None
            };
            let wind_size = if let Some(dict) = dict {
                dict_usize(dict, "wind_size", 3)?
            } else {
                3
            };
            let velocity_texture_threshold = if let Some(dict) = dict {
                dict_f32(dict, "velocity_texture_threshold", 4.0)?
            } else {
                4.0
            };
            let reflectivity_threshold = if let Some(dict) = dict {
                dict_f32(dict, "reflectivity_threshold", 0.0)?
            } else {
                0.0
            };
            let interval_splits = if let Some(dict) = dict {
                dict_usize(dict, "interval_splits", 3)?
            } else {
                3
            };
            let skip_between_rays = if let Some(dict) = dict {
                dict_usize(dict, "skip_between_rays", 100)?
            } else {
                100
            };
            let skip_along_ray = if let Some(dict) = dict {
                dict_usize(dict, "skip_along_ray", 100)?
            } else {
                100
            };
            let centered = if let Some(dict) = dict {
                dict_bool(dict, "centered", true)?
            } else {
                true
            };
            let rays_wrap_around = if let Some(dict) = dict {
                dict_bool(dict, "rays_wrap_around", true)?
            } else {
                true
            };
            let fill_value = if let Some(dict) = dict {
                dict_opt_f32(dict, "fill_value")?
            } else {
                Some(-64.5)
            };
            let fill_tolerance = if let Some(dict) = dict {
                dict_f32(dict, "fill_tolerance", 1.0)?
            } else {
                1.0
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "vradh_winding_number")?
            } else {
                "vradh_winding_number".to_string()
            };
            Ok(QcOp::VradhWindingNumber {
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
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown QC step: {}",
            name
        ))),
    }
}

fn build_qc_outputs(raystack: &RaystackData, qc_ops: &[QcOp]) -> Vec<(String, QcArray)> {
    let mut outputs = Vec::new();
    for op in qc_ops {
        match op {
            QcOp::RhohvThreshold { threshold, vname } => {
                let mask = qc::rhohv_threshold(&raystack.rhohv, *threshold);
                outputs.push((vname.clone(), QcArray::Mask(mask)));
            }
            QcOp::SunSpike {
                dbzh_threshold,
                fill_threshold,
                corr_threshold,
                vname,
            } => {
                let mask = qc::sun_spike(
                    &raystack.dbzh,
                    raystack.n_radials,
                    raystack.fold_size,
                    *dbzh_threshold,
                    *fill_threshold,
                    *corr_threshold,
                );
                outputs.push((vname.clone(), QcArray::Mask(mask)));
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
                let mut out = vec![f32::NAN; raystack.n_radials * raystack.fold_size];
                for sweep in &raystack.sweeps {
                    let start = sweep.start_index;
                    let n_radials = sweep.n_radials;
                    let fold_size = raystack.fold_size;
                    let slice_len = n_radials * fold_size;
                    let offset = start * fold_size;
                    let vradh = &raystack.vradh[offset..offset + slice_len];
                    let dbzh = &raystack.dbzh[offset..offset + slice_len];
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
                    let sweep_out =
                        qc::vradh_winding_number(vradh, Some(dbzh), n_radials, fold_size, params);
                    out[offset..offset + slice_len].copy_from_slice(&sweep_out);
                }
                outputs.push((vname.clone(), QcArray::Float(out)));
            }
        }
    }
    outputs
}

/// Parse NEXRAD data directly to raystack format (optimized)
///
/// Uses two-pass parsing:
/// 1. First pass: collect metadata and count radials
/// 2. Preallocate arrays
/// 3. Second pass: decode data into preallocated buffers
pub fn parse_optimized(data: &[u8], fold_size: usize) -> Result<RaystackData> {
    // Handle outer gzip
    let data = ungzip_if_needed(data)?;

    // Parse using nexrad-data (handles BZ2 decompression internally with parallel feature)
    let volume = VolumeFile::new(data.into_owned());
    let meta_info = extract_scan_meta(&volume);

    let scan = volume.scan()?;

    // First pass: collect metadata
    let meta = collect_metadata(&scan);

    // Allocate output
    let mut raystack = RaystackData::with_capacity(&meta, fold_size, &meta_info);

    // Second pass: fill data (pass sweep metadata for max_gates per sweep)
    fill_raystack_data(&scan, &mut raystack, &meta.sweeps);

    Ok(raystack)
}

/// First pass: collect metadata without allocating moment data
fn collect_metadata(scan: &Scan) -> VolumeMeta {
    let mut sweeps = Vec::new();
    let mut total_radials = 0;

    for sweep in scan.sweeps() {
        let radials = sweep.radials();
        let n_radials = radials.len();
        if n_radials == 0 {
            continue;
        }

        let first_radial = &radials[0];

        // Find max gate count across all moments in this sweep.
        // Use that moment's range metadata to define the sweep grid (matches xradar behavior).
        let mut max_gates = 0usize;
        let mut range_first_km = 0.0f64;
        let mut gate_interval_km = 0.0f64;

        let mut update_grid = |moment: Option<&nexrad_model::data::MomentData>| {
            if let Some(m) = moment {
                let gates = m.gate_count() as usize;
                if gates > max_gates {
                    max_gates = gates;
                    range_first_km = m.first_gate_range_km();
                    gate_interval_km = m.gate_interval_km();
                }
            }
        };

        for radial in radials {
            update_grid(radial.reflectivity());
            update_grid(radial.velocity());
            update_grid(radial.spectrum_width());
            update_grid(radial.differential_reflectivity());
            update_grid(radial.differential_phase());
            update_grid(radial.correlation_coefficient());
            update_grid(radial.specific_differential_phase());
        }

        sweeps.push(SweepMeta {
            elevation_number: sweep.elevation_number(),
            elevation_angle: first_radial.elevation_angle_degrees(),
            n_radials,
            max_gates,
            range_first_km,
            gate_interval_km,
        });

        total_radials += n_radials;
    }

    VolumeMeta {
        pattern_number: scan.coverage_pattern_number(),
        sweeps,
        total_radials,
    }
}

/// Second pass: fill preallocated buffers
fn fill_raystack_data(scan: &Scan, raystack: &mut RaystackData, sweep_meta: &[SweepMeta]) {
    let fold_size = raystack.fold_size;
    let mut radial_idx = 0;
    let mut meta_idx = 0;
    let mut sweep_counter: u32 = 0;

    for sweep in scan.sweeps().iter() {
        let radials = sweep.radials();
        if radials.is_empty() {
            continue;
        }

        // Get sweep grid metadata (computed in first pass)
        let sweep_info = sweep_meta[meta_idx];
        meta_idx += 1;

        for radial in radials {
            // Fill coordinates
            raystack.azimuth[radial_idx] = radial.azimuth_angle_degrees();
            raystack.elevation[radial_idx] = radial.elevation_angle_degrees();
            raystack.time[radial_idx] = radial.collection_timestamp();
            // Use sweep_counter to stay aligned with sweep_meta (which skips empty sweeps)
            raystack.sweep_idx[radial_idx] = sweep_counter;

            // Fill moments with folding, using sweep grid metadata for physical alignment
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_DBZH,
                radial.reflectivity(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_VRADH,
                radial.velocity(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_WRADH,
                radial.spectrum_width(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_ZDR,
                radial.differential_reflectivity(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_PHIDP,
                radial.differential_phase(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_RHOHV,
                radial.correlation_coefficient(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );
            fill_moment(
                raystack,
                radial_idx,
                MOMENT_KDP,
                radial.specific_differential_phase(),
                fold_size,
                sweep_info.max_gates,
                sweep_info.range_first_km,
                sweep_info.gate_interval_km,
            );

            radial_idx += 1;
        }

        sweep_counter += 1;
    }
}

/// Fill a moment array for one radial with range folding
#[inline]
fn fill_moment(
    raystack: &mut RaystackData,
    radial_idx: usize,
    moment_idx: usize,
    moment: Option<&nexrad_model::data::MomentData>,
    fold_size: usize,
    sweep_gates: usize,
    sweep_first_km: f64,
    sweep_gate_interval_km: f64,
) {
    let dest = raystack.moment_slice_mut(moment_idx, radial_idx);

    if let Some(m) = moment {
        let values = m.values();
        // Convert to f32 and fold into destination
        // Use sweep grid metadata for physical range alignment
        fold_moment_values_into(
            &values,
            dest,
            fold_size,
            sweep_gates,
            m.first_gate_range_km(),
            m.gate_interval_km(),
            sweep_first_km,
            sweep_gate_interval_km,
        );
    }
    // If None, dest is already filled with NaN from initialization
}

/// Fold moment values directly into destination buffer
///
/// Uses `max_gates` as the common range grid size for all moments in the radial.
/// This ensures that moments with different native gate counts are mapped to
/// the same physical ranges when folded (matching xradar's behavior).
#[inline]
fn fold_moment_values_into(
    values: &[MomentValue],
    dest: &mut [f32],
    fold_size: usize,
    sweep_gates: usize,
    moment_first_km: f64,
    moment_gate_interval_km: f64,
    sweep_first_km: f64,
    sweep_gate_interval_km: f64,
) {
    let n_gates = values.len();
    if n_gates == 0 || sweep_gates == 0 || fold_size == 0 {
        return; // dest already NaN
    }

    if moment_gate_interval_km <= 0.0 || sweep_gate_interval_km <= 0.0 {
        return;
    }

    let same_grid = (moment_first_km - sweep_first_km).abs() < 1e-6
        && (moment_gate_interval_km - sweep_gate_interval_km).abs() < 1e-6;

    if same_grid {
        if sweep_gates <= fold_size {
            // No folding needed, just copy (padding with NaN beyond n_gates)
            for i in 0..fold_size.min(n_gates) {
                dest[i] = match values[i] {
                    MomentValue::Value(x) => x,
                    _ => f32::NAN,
                };
            }
            // Rest of dest stays NaN from initialization
        } else {
            // Fold: average values into buckets based on sweep_gates (common range grid)
            let bucket_size = sweep_gates as f32 / fold_size as f32;

            for i in 0..fold_size {
                let start = (i as f32 * bucket_size) as usize;
                let end = ((i + 1) as f32 * bucket_size) as usize;
                let end = end.min(sweep_gates);

                let mut sum = 0.0f32;
                let mut count = 0u32;

                // Only process gates that exist in this moment's data
                // Gates beyond n_gates are treated as NaN (don't contribute)
                for j in start..end.min(n_gates) {
                    if let MomentValue::Value(x) = values[j] {
                        sum += x;
                        count += 1;
                    }
                }

                dest[i] = if count > 0 {
                    sum / count as f32
                } else {
                    f32::NAN
                };
            }
        }
        return;
    }

    if sweep_gates <= fold_size {
        // Map moment gates to sweep grid indices by physical range
        let mut sum = vec![0.0f32; sweep_gates];
        let mut count = vec![0u32; sweep_gates];

        for (gate_idx, value) in values.iter().enumerate() {
            let val = match value {
                MomentValue::Value(x) => *x,
                _ => continue,
            };

            let range_km = moment_first_km + gate_idx as f64 * moment_gate_interval_km;
            let sweep_pos = (range_km - sweep_first_km) / sweep_gate_interval_km;
            if sweep_pos < 0.0 || sweep_pos >= sweep_gates as f64 {
                continue;
            }
            let idx = sweep_pos.round() as isize;
            if idx < 0 || idx >= sweep_gates as isize {
                continue;
            }
            let uidx = idx as usize;
            sum[uidx] += val;
            count[uidx] += 1;
        }

        for i in 0..sweep_gates {
            if count[i] > 0 {
                dest[i] = sum[i] / count[i] as f32;
            }
        }
    } else {
        // Fold using physical range mapping into sweep grid buckets
        let mut sum = vec![0.0f32; fold_size];
        let mut count = vec![0u32; fold_size];
        let bucket_size = sweep_gates as f64 / fold_size as f64;

        for (gate_idx, value) in values.iter().enumerate() {
            let val = match value {
                MomentValue::Value(x) => *x,
                _ => continue,
            };

            let range_km = moment_first_km + gate_idx as f64 * moment_gate_interval_km;
            let sweep_pos = (range_km - sweep_first_km) / sweep_gate_interval_km;
            if sweep_pos < 0.0 || sweep_pos >= sweep_gates as f64 {
                continue;
            }
            let bucket = (sweep_pos / bucket_size).floor() as isize;
            if bucket < 0 || bucket >= fold_size as isize {
                continue;
            }

            let ubucket = bucket as usize;
            sum[ubucket] += val;
            count[ubucket] += 1;
        }

        for i in 0..fold_size {
            dest[i] = if count[i] > 0 {
                sum[i] / count[i] as f32
            } else {
                f32::NAN
            };
        }
    }
}

/// Parse NEXRAD data to raystack format (Python wrapper)
#[pyfunction]
#[pyo3(name = "parse", signature = (data, fold_size = None, qc = None, include_activity = true))]
pub fn parse_py<'py>(
    py: Python<'py>,
    data: &[u8],
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;
    let bytes_len = data.len();
    let start = Instant::now();

    // Release GIL during parsing
    let raystack = match py.detach(|| parse_optimized(data, fold_size)) {
        Ok(raystack) => raystack,
        Err(err) => {
            tracing::warn!(
                target: "radrs::parse",
                format = "raystack",
                bytes = bytes_len,
                fold_size,
                error = %err
            );
            return Err(err.into());
        }
    };
    let elapsed = start.elapsed();
    tracing::info!(
        target: "radrs::parse",
        format = "raystack",
        bytes = bytes_len,
        fold_size,
        n_radials = raystack.n_radials,
        n_sweeps = raystack.sweeps.len(),
        elapsed_ms = elapsed.as_millis() as u64
    );

    raystack_to_python(py, raystack, &qc_ops, include_activity)
}

/// Open a NEXRAD Level 2 file and return raystack DataTree
#[pyfunction]
#[pyo3(name = "open_datatree", signature = (source, fold_size = None, qc = None, include_activity = true))]
pub fn open_raystack_datatree_py<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;

    let data = if source.is_instance_of::<PyBytes>() {
        source.extract::<Vec<u8>>()?
    } else {
        let path_str: String = source.extract()?;
        if path_str.starts_with("s3://") {
            RUNTIME.block_on(fetch_s3_url(&path_str))?
        } else {
            fs::read(&path_str).map_err(RadrsError::Io)?
        }
    };

    let bytes_len = data.len();
    let start = Instant::now();
    let raystack = match py.detach(|| parse_optimized(&data, fold_size)) {
        Ok(raystack) => raystack,
        Err(err) => {
            tracing::warn!(
                target: "radrs::parse",
                format = "raystack",
                bytes = bytes_len,
                fold_size,
                error = %err
            );
            return Err(err.into());
        }
    };
    let elapsed = start.elapsed();
    tracing::info!(
        target: "radrs::parse",
        format = "raystack",
        bytes = bytes_len,
        fold_size,
        n_radials = raystack.n_radials,
        n_sweeps = raystack.sweeps.len(),
        elapsed_ms = elapsed.as_millis() as u64
    );
    raystack_data_to_raystack_datatree(py, raystack, &qc_ops, include_activity)
}

/// Open a NEXRAD Level 2 file asynchronously and return raystack DataTree
#[pyfunction]
#[pyo3(name = "open_datatree_async", signature = (source, fold_size = None, qc = None, include_activity = true))]
pub fn open_raystack_datatree_async_py<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let source = source.as_borrowed().to_owned().unbind();
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;

    let awaitable = future_into_py(py, async move {
        let data = fetch_source_bytes_async(source).await?;

        let bytes_len = data.len();
        let start = Instant::now();
        let raystack = tokio::task::spawn_blocking(move || parse_optimized(&data, fold_size))
            .await
            .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;
        let elapsed = start.elapsed();
        tracing::info!(
            target: "radrs::parse",
            format = "raystack",
            bytes = bytes_len,
            fold_size,
            n_radials = raystack.n_radials,
            n_sweeps = raystack.sweeps.len(),
            elapsed_ms = elapsed.as_millis() as u64
        );

        Python::attach(|py| {
            raystack_data_to_raystack_datatree(py, raystack, &qc_ops, include_activity)
        })
        .map_err(Into::into)
    })?;

    Ok(awaitable.into())
}

async fn fetch_source_bytes_async(source: Py<PyAny>) -> Result<Vec<u8>> {
    enum Source {
        Bytes(Vec<u8>),
        Path(String),
    }

    let source = Python::attach(|py| -> Result<Source> {
        let obj = source.bind(py);
        if obj.is_instance_of::<PyBytes>() {
            let bytes = obj
                .extract::<Vec<u8>>()
                .map_err(|e: PyErr| RadrsError::Python(e.to_string()))?;
            Ok(Source::Bytes(bytes))
        } else {
            let path = obj
                .extract::<String>()
                .map_err(|e: PyErr| RadrsError::Python(e.to_string()))?;
            Ok(Source::Path(path))
        }
    })?;

    match source {
        Source::Bytes(bytes) => Ok(bytes),
        Source::Path(path) => {
            if path.starts_with("s3://") {
                fetch_s3_url(&path).await
            } else {
                tokio::task::spawn_blocking(move || fs::read(&path))
                    .await
                    .map_err(|e| RadrsError::Python(format!("File read task failed: {}", e)))?
                    .map_err(RadrsError::Io)
            }
        }
    }
}

fn raystack_data_to_raystack_datatree(
    py: Python<'_>,
    raystack: RaystackData,
    qc_ops: &[QcOp],
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    let n_returns = raystack.n_radials;
    let fold_size = raystack.fold_size;
    let instrument_name = raystack.instrument_name.clone();
    let qc_outputs = if !qc_ops.is_empty() && n_returns > 0 {
        build_qc_outputs(&raystack, qc_ops)
    } else {
        Vec::new()
    };
    let activity = if include_activity {
        Some(compute_activity(&raystack))
    } else {
        None
    };

    let vcp_time = raystack.time.iter().copied().min().unwrap_or(0);
    let vcp_end = raystack.time.iter().copied().max().unwrap_or(vcp_time);
    let vcp_duration_ms = vcp_end - vcp_time;

    let mut sweep_times: Vec<i64> = Vec::with_capacity(raystack.sweeps.len());
    let mut sweep_durations: Vec<i64> = Vec::with_capacity(raystack.sweeps.len());
    let mut sweep_time_per_return = vec![vcp_time; n_returns];
    let mut sweep_number_per_return = vec![0u32; n_returns];
    let mut elevation_number_per_return = vec![0u32; n_returns];
    let mut base_range_per_return = vec![f32::NAN; n_returns];
    let mut range_step_per_return = vec![f32::NAN; n_returns];

    let mut sweep_numbers: Vec<u32> = Vec::with_capacity(raystack.sweeps.len());
    let mut elevation_numbers: Vec<u32> = Vec::with_capacity(raystack.sweeps.len());
    let mut elevation_angles: Vec<f32> = Vec::with_capacity(raystack.sweeps.len());
    let mut n_radials_list: Vec<u32> = Vec::with_capacity(raystack.sweeps.len());
    let mut start_indices: Vec<u32> = Vec::with_capacity(raystack.sweeps.len());
    let mut range_starts: Vec<f32> = Vec::with_capacity(raystack.sweeps.len());
    let mut range_steps: Vec<f32> = Vec::with_capacity(raystack.sweeps.len());
    let mut max_ranges: Vec<f32> = Vec::with_capacity(raystack.sweeps.len());

    for sweep in &raystack.sweeps {
        let start = sweep.start_index;
        let sweep_time = raystack.time.get(start).copied().unwrap_or(vcp_time);
        sweep_times.push(sweep_time);
        let end = (start + sweep.n_radials).min(n_returns);
        let sweep_end = raystack
            .time
            .get(end.saturating_sub(1))
            .copied()
            .unwrap_or(sweep_time);
        sweep_durations.push(sweep_end - sweep_time);

        let (range_start_m, range_step_m, max_range_m) = sweep_range_metrics(sweep, fold_size);

        sweep_numbers.push(sweep.sweep_number);
        elevation_numbers.push(u32::from(sweep.elevation_number));
        elevation_angles.push(sweep.elevation_angle);
        n_radials_list.push(sweep.n_radials as u32);
        start_indices.push(sweep.start_index as u32);
        range_starts.push(range_start_m);
        range_steps.push(range_step_m);
        max_ranges.push(max_range_m);

        for idx in start..end {
            sweep_time_per_return[idx] = sweep_time;
            sweep_number_per_return[idx] = sweep.sweep_number;
            elevation_number_per_return[idx] = u32::from(sweep.elevation_number);
            base_range_per_return[idx] = range_start_m;
            range_step_per_return[idx] = range_step_m;
        }
    }

    let vcp_time_arr = np.call_method1("array", (&vec![vcp_time],))?;
    let vcp_time_dt = vcp_time_arr.call_method1("astype", ("datetime64[ms]",))?;
    let vcp_duration_arr = np.call_method1("array", (&vec![vcp_duration_ms],))?;
    let vcp_duration_dt = vcp_duration_arr.call_method1("astype", ("timedelta64[ms]",))?;

    let sweep_time_arr = np.call_method1("array", (&sweep_times,))?;
    let sweep_time_dt = sweep_time_arr.call_method1("astype", ("datetime64[ms]",))?;
    let sweep_duration_arr = np.call_method1("array", (&sweep_durations,))?;
    let sweep_duration_dt = sweep_duration_arr.call_method1("astype", ("timedelta64[ms]",))?;

    let return_time_arr = raystack.time.into_pyarray(py);
    let return_time_dt = np.call_method1("array", (&return_time_arr,))?;
    let return_time_dt = return_time_dt.call_method1("astype", ("datetime64[ms]",))?;

    let sweep_time_per_return_arr = sweep_time_per_return.into_pyarray(py);
    let sweep_time_per_return_dt = np.call_method1("array", (&sweep_time_per_return_arr,))?;
    let sweep_time_per_return_dt =
        sweep_time_per_return_dt.call_method1("astype", ("datetime64[ms]",))?;

    // vcps dataset
    let vcps_coords = PyDict::new(py);
    vcps_coords.set_item("vcp_time", (("vcp_time",), vcp_time_dt.clone()))?;
    if let Some(ref name) = instrument_name {
        vcps_coords.set_item("instrument_name", (("vcp_time",), vec![name.as_str()]))?;
    }

    let vcps_vars = PyDict::new(py);
    vcps_vars.set_item(
        "pattern_number",
        (("vcp_time",), vec![u32::from(raystack.pattern_number)]),
    )?;
    vcps_vars.set_item(
        "vcp_name",
        (
            ("vcp_time",),
            vec![format!("VCP-{}", raystack.pattern_number)],
        ),
    )?;
    vcps_vars.set_item("vcp_duration", (("vcp_time",), vcp_duration_dt))?;
    vcps_vars.set_item(
        "num_sweeps",
        (("vcp_time",), vec![raystack.sweeps.len() as u32]),
    )?;
    vcps_vars.set_item("instrument_type", (("vcp_time",), vec![INSTRUMENT_TYPE]))?;
    vcps_vars.set_item("platform_type", (("vcp_time",), vec![PLATFORM_TYPE]))?;
    if let Some(lat) = raystack.latitude {
        vcps_vars.set_item("latitude", (("vcp_time",), vec![lat]))?;
    }
    if let Some(lon) = raystack.longitude {
        vcps_vars.set_item("longitude", (("vcp_time",), vec![lon]))?;
    }
    if let Some(alt) = raystack.altitude {
        vcps_vars.set_item("altitude", (("vcp_time",), vec![alt]))?;
    }

    let vcps_ds = xr.call_method(
        "Dataset",
        (),
        Some(
            &[
                ("data_vars", vcps_vars.as_any()),
                ("coords", vcps_coords.as_any()),
            ]
            .into_py_dict(py)?,
        ),
    )?;

    // sweeps dataset
    let sweeps_coords = PyDict::new(py);
    sweeps_coords.set_item("sweep_time", (("sweep_time",), sweep_time_dt.clone()))?;
    sweeps_coords.set_item(
        "vcp_time",
        (("sweep_time",), vec![vcp_time; raystack.sweeps.len()]),
    )?;
    if let Some(ref name) = instrument_name {
        sweeps_coords.set_item(
            "instrument_name",
            (("sweep_time",), vec![name.as_str(); raystack.sweeps.len()]),
        )?;
    }
    if let Some(lat) = raystack.latitude {
        sweeps_coords.set_item(
            "latitude",
            (("sweep_time",), vec![lat; raystack.sweeps.len()]),
        )?;
    }
    if let Some(lon) = raystack.longitude {
        sweeps_coords.set_item(
            "longitude",
            (("sweep_time",), vec![lon; raystack.sweeps.len()]),
        )?;
    }
    if let Some(alt) = raystack.altitude {
        sweeps_coords.set_item(
            "altitude",
            (("sweep_time",), vec![alt; raystack.sweeps.len()]),
        )?;
    }

    let sweeps_vars = PyDict::new(py);
    sweeps_vars.set_item("sweep_number", (("sweep_time",), sweep_numbers))?;
    sweeps_vars.set_item("elevation_number", (("sweep_time",), elevation_numbers))?;
    sweeps_vars.set_item("sweep_fixed_angle", (("sweep_time",), elevation_angles))?;
    sweeps_vars.set_item("n_radials", (("sweep_time",), n_radials_list))?;
    sweeps_vars.set_item("start_index", (("sweep_time",), start_indices))?;
    set_sweep_mode_vars(&sweeps_vars, raystack.sweeps.len())?;
    sweeps_vars.set_item("sweep_duration", (("sweep_time",), sweep_duration_dt))?;
    sweeps_vars.set_item("max_range", (("sweep_time",), max_ranges))?;
    sweeps_vars.set_item("range_start", (("sweep_time",), range_starts))?;
    sweeps_vars.set_item("range_step", (("sweep_time",), range_steps))?;

    let sweeps_ds = xr.call_method(
        "Dataset",
        (),
        Some(
            &[
                ("data_vars", sweeps_vars.as_any()),
                ("coords", sweeps_coords.as_any()),
            ]
            .into_py_dict(py)?,
        ),
    )?;

    // returns dataset
    let returns_coords = PyDict::new(py);
    returns_coords.set_item("return_time", (("return_time",), return_time_dt.clone()))?;
    returns_coords.set_item(
        "sweep_time",
        (("return_time",), sweep_time_per_return_dt.clone()),
    )?;
    returns_coords.set_item("vcp_time", (("return_time",), vec![vcp_time; n_returns]))?;
    if let Some(ref name) = instrument_name {
        returns_coords.set_item(
            "instrument_name",
            (("return_time",), vec![name.as_str(); n_returns]),
        )?;
    }
    if let Some(lat) = raystack.latitude {
        returns_coords.set_item("latitude", (("return_time",), vec![lat; n_returns]))?;
    }
    if let Some(lon) = raystack.longitude {
        returns_coords.set_item("longitude", (("return_time",), vec![lon; n_returns]))?;
    }
    if let Some(alt) = raystack.altitude {
        returns_coords.set_item("altitude", (("return_time",), vec![alt; n_returns]))?;
    }
    returns_coords.set_item(
        "range",
        (("range",), (0..fold_size).collect::<Vec<usize>>()),
    )?;
    returns_coords.set_item(
        "azimuth",
        (("return_time",), raystack.azimuth.into_pyarray(py)),
    )?;
    returns_coords.set_item(
        "elevation",
        (("return_time",), raystack.elevation.into_pyarray(py)),
    )?;
    returns_coords.set_item(
        "sweep_idx",
        (("return_time",), raystack.sweep_idx.into_pyarray(py)),
    )?;
    returns_coords.set_item(
        "sweep_number",
        (("return_time",), sweep_number_per_return.into_pyarray(py)),
    )?;
    returns_coords.set_item(
        "elevation_number",
        (
            ("return_time",),
            elevation_number_per_return.into_pyarray(py),
        ),
    )?;
    returns_coords.set_item(
        "base_range",
        (("return_time",), base_range_per_return.into_pyarray(py)),
    )?;
    returns_coords.set_item(
        "range_step",
        (("return_time",), range_step_per_return.into_pyarray(py)),
    )?;

    let returns_vars = PyDict::new(py);
    let add_moment = |name: &str, data: Vec<f32>, vars: &Bound<'_, PyDict>| -> PyResult<()> {
        if n_returns > 0 {
            let arr = Array2::from_shape_vec((n_returns, fold_size), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            vars.set_item(name, (("return_time", "range"), arr.into_pyarray(py)))?;
        }
        Ok(())
    };

    let moments: [(&str, Vec<f32>); 7] = [
        (MOMENT_NAMES[MOMENT_DBZH], raystack.dbzh),
        (MOMENT_NAMES[MOMENT_VRADH], raystack.vradh),
        (MOMENT_NAMES[MOMENT_WRADH], raystack.wradh),
        (MOMENT_NAMES[MOMENT_ZDR], raystack.zdr),
        (MOMENT_NAMES[MOMENT_PHIDP], raystack.phidp),
        (MOMENT_NAMES[MOMENT_RHOHV], raystack.rhohv),
        (MOMENT_NAMES[MOMENT_KDP], raystack.kdp),
    ];
    for (name, data) in moments {
        add_moment(name, data, &returns_vars)?;
    }

    let returns_ds = xr.call_method(
        "Dataset",
        (),
        Some(
            &[
                ("data_vars", returns_vars.as_any()),
                ("coords", returns_coords.as_any()),
            ]
            .into_py_dict(py)?,
        ),
    )?;

    let qc_ds = if !qc_outputs.is_empty() && n_returns > 0 {
        let qc_coords = PyDict::new(py);
        qc_coords.set_item("return_time", (("return_time",), return_time_dt.clone()))?;
        qc_coords.set_item(
            "range",
            (("range",), (0..fold_size).collect::<Vec<usize>>()),
        )?;
        qc_coords.set_item(
            "sweep_time",
            (("return_time",), sweep_time_per_return_dt.clone()),
        )?;
        qc_coords.set_item("vcp_time", (("return_time",), vec![vcp_time; n_returns]))?;

        let qc_vars = PyDict::new(py);
        for (name, output) in qc_outputs {
            match output {
                QcArray::Mask(mask) => {
                    let arr = Array2::from_shape_vec((n_returns, fold_size), mask)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    qc_vars.set_item(name, (("return_time", "range"), arr.into_pyarray(py)))?;
                }
                QcArray::Float(data) => {
                    let arr = Array2::from_shape_vec((n_returns, fold_size), data)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    qc_vars.set_item(name, (("return_time", "range"), arr.into_pyarray(py)))?;
                }
            }
        }

        Some(
            xr.call_method(
                "Dataset",
                (),
                Some(
                    &[
                        ("data_vars", qc_vars.as_any()),
                        ("coords", qc_coords.as_any()),
                    ]
                    .into_py_dict(py)?,
                ),
            )?,
        )
    } else {
        None
    };

    let activity_ds = if let Some(activity) = activity {
        let activity_coords = PyDict::new(py);
        activity_coords.set_item("moment", (("moment",), activity.moments.clone()))?;
        activity_coords.set_item("return_time", (("return_time",), return_time_dt.clone()))?;
        activity_coords.set_item("sweep_time", (("sweep_time",), sweep_time_dt.clone()))?;
        activity_coords.set_item("vcp_time", (("vcp_time",), vcp_time_dt.clone()))?;

        let activity_vars = PyDict::new(py);
        let ray_counts = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_returns),
            activity.ray_valid_count,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "ray_valid_count",
            (("moment", "return_time"), ray_counts.into_pyarray(py)),
        )?;
        let ray_frac = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_returns),
            activity.ray_valid_fraction,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "ray_valid_fraction",
            (("moment", "return_time"), ray_frac.into_pyarray(py)),
        )?;

        let sweep_counts = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_sweeps),
            activity.sweep_valid_count,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "sweep_valid_count",
            (("moment", "sweep_time"), sweep_counts.into_pyarray(py)),
        )?;
        let sweep_frac = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_sweeps),
            activity.sweep_valid_fraction,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "sweep_valid_fraction",
            (("moment", "sweep_time"), sweep_frac.into_pyarray(py)),
        )?;

        let volume_counts =
            Array2::from_shape_vec((activity.moments.len(), 1), activity.volume_valid_count)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "volume_valid_count",
            (("moment", "vcp_time"), volume_counts.into_pyarray(py)),
        )?;
        let volume_frac =
            Array2::from_shape_vec((activity.moments.len(), 1), activity.volume_valid_fraction)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_vars.set_item(
            "volume_valid_fraction",
            (("moment", "vcp_time"), volume_frac.into_pyarray(py)),
        )?;

        Some(
            xr.call_method(
                "Dataset",
                (),
                Some(
                    &[
                        ("data_vars", activity_vars.as_any()),
                        ("coords", activity_coords.as_any()),
                    ]
                    .into_py_dict(py)?,
                ),
            )?,
        )
    } else {
        None
    };

    let root_attrs = build_root_attrs(
        py,
        raystack.pattern_number,
        instrument_name.as_deref(),
    )?;

    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;

    let tree_dict = PyDict::new(py);
    tree_dict.set_item("/", root_ds)?;
    tree_dict.set_item("vcps", vcps_ds)?;
    tree_dict.set_item("sweeps", sweeps_ds)?;
    tree_dict.set_item("returns", returns_ds)?;
    if let Some(qc_ds) = qc_ds {
        tree_dict.set_item("qc", qc_ds)?;
    }
    if let Some(activity_ds) = activity_ds {
        tree_dict.set_item("activity", activity_ds)?;
    }

    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.into())
}

/// Convert RaystackData to Python dict with numpy arrays
pub fn raystack_to_python(
    py: Python<'_>,
    raystack: RaystackData,
    qc_ops: &[QcOp],
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    let activity = if include_activity {
        Some(compute_activity(&raystack))
    } else {
        None
    };

    // VCPs dict
    let vcps = PyDict::new(py);
    vcps.set_item("pattern_number", raystack.pattern_number)?;
    if let Some(ref name) = raystack.instrument_name {
        vcps.set_item("instrument_name", name.as_str())?;
    }
    vcps.set_item("vcp_name", format!("VCP-{}", raystack.pattern_number))?;
    let vcp_time = raystack.time.iter().copied().min().unwrap_or(0);
    let vcp_end = raystack.time.iter().copied().max().unwrap_or(vcp_time);
    vcps.set_item("vcp_duration", vcp_end - vcp_time)?;
    vcps.set_item("num_sweeps", raystack.sweeps.len() as u32)?;
    vcps.set_item("instrument_type", INSTRUMENT_TYPE)?;
    vcps.set_item("platform_type", PLATFORM_TYPE)?;
    if let Some(lat) = raystack.latitude {
        vcps.set_item("latitude", lat)?;
    }
    if let Some(lon) = raystack.longitude {
        vcps.set_item("longitude", lon)?;
    }
    if let Some(alt) = raystack.altitude {
        vcps.set_item("altitude", alt)?;
    }
    result.set_item("vcps", vcps)?;

    // Sweeps list
    let sweeps_list: Vec<_> = raystack
        .sweeps
        .iter()
        .map(|s| {
            let d = PyDict::new(py);
            d.set_item("sweep_number", s.sweep_number).ok();
            d.set_item("elevation_number", s.elevation_number).ok();
            d.set_item("elevation_angle", s.elevation_angle).ok();
            d.set_item("n_radials", s.n_radials).ok();
            d.set_item("start_index", s.start_index).ok();
            d.set_item("max_gates", s.max_gates).ok();
            d.set_item("range_first_km", s.range_first_km).ok();
            d.set_item("gate_interval_km", s.gate_interval_km).ok();
            d
        })
        .collect();
    result.set_item("sweeps", sweeps_list)?;

    let n_radials = raystack.n_radials;
    let fold_size = raystack.fold_size;
    let qc_outputs = if !qc_ops.is_empty() && n_radials > 0 {
        build_qc_outputs(&raystack, qc_ops)
    } else {
        Vec::new()
    };
    let returns = PyDict::new(py);

    // Coordinate arrays (1D)
    returns.set_item("azimuth", raystack.azimuth.into_pyarray(py))?;
    returns.set_item("elevation", raystack.elevation.into_pyarray(py))?;
    returns.set_item("time", raystack.time.into_pyarray(py))?;
    returns.set_item("sweep_idx", raystack.sweep_idx.into_pyarray(py))?;
    returns.set_item("range", (0..fold_size).collect::<Vec<usize>>())?;

    let mut sweep_number_per_return = vec![0u32; n_radials];
    let mut elevation_number_per_return = vec![0u32; n_radials];
    let mut base_range_per_return = vec![f32::NAN; n_radials];
    let mut range_step_per_return = vec![f32::NAN; n_radials];

    for sweep in &raystack.sweeps {
        let start = sweep.start_index;
        let end = (start + sweep.n_radials).min(n_radials);
        let (range_start_m, range_step_m, _max_range_m) = sweep_range_metrics(sweep, fold_size);

        for idx in start..end {
            sweep_number_per_return[idx] = sweep.sweep_number;
            elevation_number_per_return[idx] = u32::from(sweep.elevation_number);
            base_range_per_return[idx] = range_start_m;
            range_step_per_return[idx] = range_step_m;
        }
    }

    returns.set_item("sweep_number", sweep_number_per_return.into_pyarray(py))?;
    returns.set_item(
        "elevation_number",
        elevation_number_per_return.into_pyarray(py),
    )?;
    returns.set_item("base_range", base_range_per_return.into_pyarray(py))?;
    returns.set_item("range_step", range_step_per_return.into_pyarray(py))?;

    // Moment arrays (2D) - reshape from flat
    let add_moment = |name: &str, data: Vec<f32>, returns: &Bound<'_, PyDict>| -> PyResult<()> {
        if n_radials > 0 {
            let arr = Array2::from_shape_vec((n_radials, fold_size), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            returns.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(())
    };

    let moments: [(&str, Vec<f32>); 7] = [
        (MOMENT_NAMES[MOMENT_DBZH], raystack.dbzh),
        (MOMENT_NAMES[MOMENT_VRADH], raystack.vradh),
        (MOMENT_NAMES[MOMENT_WRADH], raystack.wradh),
        (MOMENT_NAMES[MOMENT_ZDR], raystack.zdr),
        (MOMENT_NAMES[MOMENT_PHIDP], raystack.phidp),
        (MOMENT_NAMES[MOMENT_RHOHV], raystack.rhohv),
        (MOMENT_NAMES[MOMENT_KDP], raystack.kdp),
    ];
    for (name, data) in moments {
        add_moment(name, data, &returns)?;
    }

    result.set_item("returns", returns)?;

    if !qc_outputs.is_empty() && n_radials > 0 {
        let qc = PyDict::new(py);
        for (name, output) in qc_outputs {
            match output {
                QcArray::Mask(mask) => {
                    let arr = Array2::from_shape_vec((n_radials, fold_size), mask)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    qc.set_item(name, arr.into_pyarray(py))?;
                }
                QcArray::Float(data) => {
                    let arr = Array2::from_shape_vec((n_radials, fold_size), data)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    qc.set_item(name, arr.into_pyarray(py))?;
                }
            }
        }
        result.set_item("qc", qc)?;
    }

    if let Some(activity) = activity {
        let activity_dict = PyDict::new(py);
        activity_dict.set_item("moment", activity.moments.clone())?;

        let ray_counts = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_returns),
            activity.ray_valid_count,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("ray_valid_count", ray_counts.into_pyarray(py))?;

        let ray_frac = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_returns),
            activity.ray_valid_fraction,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("ray_valid_fraction", ray_frac.into_pyarray(py))?;

        let sweep_counts = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_sweeps),
            activity.sweep_valid_count,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("sweep_valid_count", sweep_counts.into_pyarray(py))?;

        let sweep_frac = Array2::from_shape_vec(
            (activity.moments.len(), activity.n_sweeps),
            activity.sweep_valid_fraction,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("sweep_valid_fraction", sweep_frac.into_pyarray(py))?;

        let volume_counts =
            Array2::from_shape_vec((activity.moments.len(), 1), activity.volume_valid_count)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("volume_valid_count", volume_counts.into_pyarray(py))?;

        let volume_frac =
            Array2::from_shape_vec((activity.moments.len(), 1), activity.volume_valid_fraction)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        activity_dict.set_item("volume_valid_fraction", volume_frac.into_pyarray(py))?;

        result.set_item("activity", activity_dict)?;
    }

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_moment_values_into_same_grid_no_fold() {
        let values = vec![
            MomentValue::Value(1.0),
            MomentValue::BelowThreshold,
            MomentValue::Value(3.0),
            MomentValue::RangeFolded,
        ];
        let mut dest = vec![f32::NAN; 6];
        // sweep_gates = 4, fold_size = 6, same grid
        fold_moment_values_into(&values, &mut dest, 6, 4, 0.5, 0.25, 0.5, 0.25);

        assert_eq!(dest[0], 1.0);
        assert!(dest[1].is_nan());
        assert_eq!(dest[2], 3.0);
        assert!(dest[3].is_nan());
        assert!(dest[4].is_nan());
        assert!(dest[5].is_nan());
    }

    #[test]
    fn test_fold_moment_values_into_same_grid_bucket_avg() {
        // 6 values folded into 3 buckets (size 2)
        // Bucket 0: [1.0, missing] -> 1.0
        // Bucket 1: [3.0, missing] -> 3.0
        // Bucket 2: [5.0, 7.0] -> 6.0
        let values = vec![
            MomentValue::Value(1.0),
            MomentValue::BelowThreshold,
            MomentValue::Value(3.0),
            MomentValue::RangeFolded,
            MomentValue::Value(5.0),
            MomentValue::Value(7.0),
        ];
        let mut dest = vec![f32::NAN; 3];
        // sweep_gates = 6, fold_size = 3, same grid
        fold_moment_values_into(&values, &mut dest, 3, 6, 0.0, 1.0, 0.0, 1.0);

        assert!((dest[0] - 1.0).abs() < 0.01);
        assert!((dest[1] - 3.0).abs() < 0.01);
        assert!((dest[2] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_fold_moment_values_into_empty() {
        let values: Vec<MomentValue> = vec![];
        let mut dest = vec![f32::NAN; 4];
        fold_moment_values_into(&values, &mut dest, 4, 4, 0.0, 1.0, 0.0, 1.0);

        assert!(dest.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_fold_moment_values_into_offset_grid_no_fold() {
        // Sweep grid: 0.5, 1.0, 1.5, 2.0 (km), fold_size >= sweep_gates
        // Moment gates at 1.0, 2.0 should land at indices 1 and 3.
        let values = vec![MomentValue::Value(10.0), MomentValue::Value(20.0)];
        let mut dest = vec![f32::NAN; 6];
        fold_moment_values_into(&values, &mut dest, 6, 4, 1.0, 1.0, 0.5, 0.5);

        assert!(dest[0].is_nan());
        assert_eq!(dest[1], 10.0);
        assert!(dest[2].is_nan());
        assert_eq!(dest[3], 20.0);
    }

    #[test]
    fn test_fold_moment_values_into_offset_grid_folded() {
        // Sweep grid length 8, fold to 4 buckets (size 2)
        // Moment gates at ranges 1,3,5 (km) map to sweep indices 1,3,5
        // Buckets: [0-2)->10, [2-4)->20, [4-6)->30, [6-8)->NaN
        let values = vec![
            MomentValue::Value(10.0),
            MomentValue::Value(20.0),
            MomentValue::Value(30.0),
        ];
        let mut dest = vec![f32::NAN; 4];
        fold_moment_values_into(&values, &mut dest, 4, 8, 1.0, 2.0, 0.0, 1.0);

        assert!((dest[0] - 10.0).abs() < 0.01);
        assert!((dest[1] - 20.0).abs() < 0.01);
        assert!((dest[2] - 30.0).abs() < 0.01);
        assert!(dest[3].is_nan());
    }
}
