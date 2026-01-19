//! Optimized direct parsing to raystack format
//!
//! Uses two-pass parsing with preallocation for minimal allocations.
//! Supports parallel BZ2 decompression via rayon.

use crate::error::Result;
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{MomentValue, Scan};
use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::borrow::Cow;
use std::io::Read;

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

/// Sweep-level metadata (public)
#[derive(Clone)]
pub struct SweepInfo {
    pub elevation_number: u8,
    pub elevation_angle: f32,
    pub n_radials: usize,
    pub start_index: usize,
}

impl RaystackData {
    /// Allocate with known sizes
    fn with_capacity(meta: &VolumeMeta, fold_size: usize) -> Self {
        let n = meta.total_radials;
        let moment_len = n * fold_size;

        let mut sweeps = Vec::with_capacity(meta.sweeps.len());
        let mut start_index = 0;
        for sm in &meta.sweeps {
            sweeps.push(SweepInfo {
                elevation_number: sm.elevation_number,
                elevation_angle: sm.elevation_angle,
                n_radials: sm.n_radials,
                start_index,
            });
            start_index += sm.n_radials;
        }

        Self {
            pattern_number: meta.pattern_number,
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
    let scan = volume.scan()?;

    // First pass: collect metadata
    let meta = collect_metadata(&scan);

    // Allocate output
    let mut raystack = RaystackData::with_capacity(&meta, fold_size);

    // Second pass: fill data
    fill_raystack_data(&scan, &mut raystack);

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

        sweeps.push(SweepMeta {
            elevation_number: sweep.elevation_number(),
            elevation_angle: first_radial.elevation_angle_degrees(),
            n_radials,
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
fn fill_raystack_data(scan: &Scan, raystack: &mut RaystackData) {
    let fold_size = raystack.fold_size;
    let mut radial_idx = 0;

    for (sweep_idx, sweep) in scan.sweeps().iter().enumerate() {
        for radial in sweep.radials() {
            // Fill coordinates
            raystack.azimuth[radial_idx] = radial.azimuth_angle_degrees();
            raystack.elevation[radial_idx] = radial.elevation_angle_degrees();
            raystack.time[radial_idx] = radial.collection_timestamp();
            raystack.sweep_idx[radial_idx] = sweep_idx as u32;

            // Fill moments with folding
            fill_moment(raystack, radial_idx, MOMENT_DBZH, radial.reflectivity(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_VRADH, radial.velocity(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_WRADH, radial.spectrum_width(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_ZDR, radial.differential_reflectivity(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_PHIDP, radial.differential_phase(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_RHOHV, radial.correlation_coefficient(), fold_size);
            fill_moment(raystack, radial_idx, MOMENT_KDP, radial.specific_differential_phase(), fold_size);

            radial_idx += 1;
        }
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
) {
    let dest = raystack.moment_slice_mut(moment_idx, radial_idx);

    if let Some(m) = moment {
        let values = m.values();
        // Convert to f32 and fold into destination
        fold_moment_values_into(&values, dest, fold_size);
    }
    // If None, dest is already filled with NaN from initialization
}

/// Fold moment values directly into destination buffer
#[inline]
fn fold_moment_values_into(
    values: &[MomentValue],
    dest: &mut [f32],
    fold_size: usize,
) {
    let n_gates = values.len();
    if n_gates == 0 {
        return; // dest already NaN
    }

    if n_gates <= fold_size {
        // No folding needed, just copy
        for (i, v) in values.iter().enumerate() {
            dest[i] = match v {
                MomentValue::Value(x) => *x,
                _ => f32::NAN,
            };
        }
    } else {
        // Fold: average values into buckets
        let bucket_size = n_gates as f32 / fold_size as f32;

        for i in 0..fold_size {
            let start = (i as f32 * bucket_size) as usize;
            let end = ((i + 1) as f32 * bucket_size) as usize;
            let end = end.min(n_gates);

            let mut sum = 0.0f32;
            let mut count = 0u32;

            for j in start..end {
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
}

/// Parse NEXRAD data to raystack format (Python wrapper)
#[pyfunction]
#[pyo3(name = "parse", signature = (data, fold_size = None, qc = None))]
pub fn parse_py(
    py: Python<'_>,
    data: &[u8],
    fold_size: Option<usize>,
    qc: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);

    // Release GIL during parsing
    let raystack = py.allow_threads(|| parse_optimized(data, fold_size))?;

    raystack_to_python(py, raystack, qc)
}

/// Convert RaystackData to Python dict with numpy arrays
pub fn raystack_to_python(
    py: Python<'_>,
    raystack: RaystackData,
    _qc: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);

    // VCPs dict
    let vcps = PyDict::new(py);
    vcps.set_item("pattern_number", raystack.pattern_number)?;
    result.set_item("vcps", vcps)?;

    // Sweeps list
    let sweeps_list: Vec<_> = raystack
        .sweeps
        .iter()
        .map(|s| {
            let d = PyDict::new(py);
            d.set_item("elevation_number", s.elevation_number).ok();
            d.set_item("elevation_angle", s.elevation_angle).ok();
            d.set_item("n_radials", s.n_radials).ok();
            d.set_item("start_index", s.start_index).ok();
            d
        })
        .collect();
    result.set_item("sweeps", sweeps_list)?;

    // Returns dict with numpy arrays
    let returns = PyDict::new(py);
    let n_radials = raystack.n_radials;
    let fold_size = raystack.fold_size;

    // Coordinate arrays (1D)
    returns.set_item("azimuth", raystack.azimuth.into_pyarray(py))?;
    returns.set_item("elevation", raystack.elevation.into_pyarray(py))?;
    returns.set_item("time", raystack.time.into_pyarray(py))?;
    returns.set_item("sweep_idx", raystack.sweep_idx.into_pyarray(py))?;

    // Moment arrays (2D) - reshape from flat
    let add_moment = |name: &str, data: Vec<f32>, returns: &Bound<'_, PyDict>| -> PyResult<()> {
        if n_radials > 0 {
            let arr = Array2::from_shape_vec((n_radials, fold_size), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            returns.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(())
    };

    add_moment("DBZH", raystack.dbzh, &returns)?;
    add_moment("VRADH", raystack.vradh, &returns)?;
    add_moment("WRADH", raystack.wradh, &returns)?;
    add_moment("ZDR", raystack.zdr, &returns)?;
    add_moment("PHIDP", raystack.phidp, &returns)?;
    add_moment("RHOHV", raystack.rhohv, &returns)?;
    add_moment("KDP", raystack.kdp, &returns)?;

    result.set_item("returns", returns)?;

    Ok(result.into())
}

