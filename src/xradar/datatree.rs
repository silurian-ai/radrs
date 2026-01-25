//! DataTree conversion for xradar compatibility

use crate::error::{RadrsError, Result};
use crate::fetch::{RUNTIME, fetch_s3_url};
use crate::metadata::{ScanMeta, extract_scan_meta};
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{MomentValue, Radial, Scan, Sweep};
use numpy::IntoPyArray;
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::fs;

/// Standard NEXRAD variable names (following xradar/CF-radial conventions)
const MOMENT_NAMES: [(&str, &str); 7] = [
    ("reflectivity", "DBZH"),
    ("velocity", "VRADH"),
    ("spectrum_width", "WRADH"),
    ("differential_reflectivity", "ZDR"),
    ("differential_phase", "PHIDP"),
    ("correlation_coefficient", "RHOHV"),
    ("specific_differential_phase", "KDP"),
];

/// Open a NEXRAD Level 2 file and return an xarray DataTree
///
/// # Arguments
/// * `source` - Path to file, S3 URL, or bytes
///
/// # Returns
/// xarray.DataTree with sweeps as children
#[pyfunction]
#[pyo3(name = "open_datatree", signature = (source, sort_by_azimuth = false))]
pub fn open_datatree_py(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    sort_by_azimuth: bool,
) -> PyResult<Py<PyAny>> {
    // Handle different input types
    let data = if source.is_instance_of::<PyBytes>() {
        // Bytes input
        source.extract::<Vec<u8>>()?
    } else {
        // String path or URL
        let path_str: String = source.extract()?;

        if path_str.starts_with("s3://") {
            // S3 URL - use async fetch
            RUNTIME.block_on(fetch_s3_url(&path_str))?
        } else {
            // Local file path
            fs::read(&path_str).map_err(RadrsError::Io)?
        }
    };

    // Parse NEXRAD data without holding the GIL
    let (scan, meta) = py.detach(|| parse_nexrad_data(data))?;

    // Convert to DataTree
    scan_to_datatree(py, &scan, &meta, sort_by_azimuth)
}

/// Open a NEXRAD Level 2 file asynchronously and return an xarray DataTree.
#[pyfunction]
#[pyo3(name = "open_datatree_async", signature = (source, sort_by_azimuth = false))]
pub fn open_datatree_async_py(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    sort_by_azimuth: bool,
) -> PyResult<Py<PyAny>> {
    let source = source.as_borrowed().to_owned().unbind();

    let awaitable = future_into_py(py, async move {
        let data = fetch_source_bytes_async(source).await?;

        let (scan, meta) = tokio::task::spawn_blocking(move || parse_nexrad_data(data))
            .await
            .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;

        Python::attach(|py| scan_to_datatree(py, &scan, &meta, sort_by_azimuth)).map_err(Into::into)
    })?;

    Ok(awaitable.into())
}

/// Open a NEXRAD file from path, URL, or bytes
pub fn open_datatree(source: Vec<u8>) -> Result<(Scan, ScanMeta)> {
    parse_nexrad_data(source)
}

/// Parse raw NEXRAD data into a Scan
fn parse_nexrad_data(data: Vec<u8>) -> Result<(Scan, ScanMeta)> {
    let volume = VolumeFile::new(data);
    let meta = extract_scan_meta(&volume);
    let scan = volume.scan()?;
    Ok((scan, meta))
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

pub(crate) fn scan_to_datatree(
    py: Python<'_>,
    scan: &Scan,
    meta: &ScanMeta,
    sort_by_azimuth: bool,
) -> PyResult<Py<PyAny>> {
    // Import xarray
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    // Build root attributes
    let root_attrs = PyDict::new(py);
    root_attrs.set_item("Conventions", "CF-1.8")?;
    root_attrs.set_item("instrument_type", "radar")?;
    root_attrs.set_item("platform_type", "fixed")?;
    root_attrs.set_item("volume_coverage_pattern", scan.coverage_pattern_number())?;
    root_attrs.set_item(
        "scan_name",
        format!("VCP-{}", scan.coverage_pattern_number()),
    )?;
    if let Some(ref name) = meta.instrument_name {
        root_attrs.set_item("instrument_name", name.as_str())?;
    }

    // Build the DataTree structure - create root with attrs
    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;

    // Create DataTree - xarray 2024+ uses from_dict
    let tree_dict = PyDict::new(py);
    tree_dict.set_item("/", root_ds)?;

    for (sweep_idx, sweep) in scan.sweeps().iter().enumerate() {
        let sweep_name = format!("/sweep_{}", sweep_idx);
        let dataset = sweep_to_dataset(py, &xr, &np, sweep, sweep_idx, meta, sort_by_azimuth)?;
        tree_dict.set_item(&sweep_name, dataset)?;
    }

    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.unbind())
}

/// Convert a Sweep to an xarray Dataset
fn sweep_to_dataset<'py>(
    py: Python<'py>,
    xr: &Bound<'py, PyModule>,
    np: &Bound<'py, PyModule>,
    sweep: &Sweep,
    sweep_idx: usize,
    meta: &ScanMeta,
    sort_by_azimuth: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let radials = sweep.radials();
    if radials.is_empty() {
        // Return empty dataset
        return xr.call_method1("Dataset", (PyDict::new(py),));
    }

    let n_rays = radials.len();

    // Create sorted indices if requested (to match xradar behavior)
    let sorted_indices: Vec<usize> = if sort_by_azimuth {
        let mut indices: Vec<usize> = (0..n_rays).collect();
        indices.sort_by(|&a, &b| {
            let az_a = radials[a].azimuth_angle_degrees();
            let az_b = radials[b].azimuth_angle_degrees();
            az_a.partial_cmp(&az_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    } else {
        (0..n_rays).collect()
    };

    // Determine range info for each moment type and find the max range
    let mut moment_info: HashMap<&str, (usize, f64, f64)> = HashMap::new();
    let mut max_n_gates = 0usize;
    let mut max_range_info: Option<(f64, f64)> = None;

    for radial in radials {
        for (moment_getter, cf_name) in &MOMENT_NAMES {
            if let Some(moment) = get_moment_data(radial, moment_getter) {
                let n_gates = moment.gate_count() as usize;
                let first_range = moment.first_gate_range_km();
                let gate_interval = moment.gate_interval_km();

                let entry = moment_info
                    .entry(*cf_name)
                    .or_insert((0, first_range, gate_interval));
                if n_gates > entry.0 {
                    *entry = (n_gates, first_range, gate_interval);
                }

                // Track the moment with the maximum gate count for range coordinate
                if n_gates > max_n_gates {
                    max_n_gates = n_gates;
                    max_range_info = Some((first_range, gate_interval));
                }
            }
        }
    }

    // If no moment data, return empty dataset
    if moment_info.is_empty() || max_range_info.is_none() {
        return xr.call_method1("Dataset", (PyDict::new(py),));
    }

    let (range_first, range_interval) = max_range_info.unwrap();

    // Build coordinate arrays (using sorted order if requested)
    let mut azimuth_data: Vec<f32> = Vec::with_capacity(n_rays);
    let mut elevation_data: Vec<f32> = Vec::with_capacity(n_rays);
    let mut time_data: Vec<i64> = Vec::with_capacity(n_rays);

    for &idx in &sorted_indices {
        let radial = &radials[idx];
        azimuth_data.push(radial.azimuth_angle_degrees());
        elevation_data.push(radial.elevation_angle_degrees());
        time_data.push(radial.collection_timestamp());
    }

    // Create data variables dict
    let data_vars = PyDict::new(py);
    let coords = PyDict::new(py);

    // Create azimuth coordinate
    let azimuth_arr = azimuth_data.into_pyarray(py);
    coords.set_item("azimuth", (("time",), azimuth_arr))?;

    // Create elevation coordinate
    let elevation_arr = elevation_data.into_pyarray(py);
    coords.set_item("elevation", (("time",), elevation_arr))?;

    // Create time coordinate (as int64 nanoseconds for datetime64[ns])
    // Convert milliseconds to datetime64[ns]
    let time_arr_i64 = time_data.into_pyarray(py);
    let time_arr = np.call_method1("array", (time_arr_i64,))?;
    let time_arr_ms = time_arr.call_method1("astype", ("datetime64[ms]",))?;
    let time_arr_ns = time_arr_ms.call_method1("astype", ("datetime64[ns]",))?;
    coords.set_item("time", (("time",), time_arr_ns))?;

    // Build range coordinate using max_n_gates (in meters)
    let range_data: Vec<f64> = (0..max_n_gates)
        .map(|i| (range_first + i as f64 * range_interval) * 1000.0)
        .collect();
    let range_arr = range_data.into_pyarray(py);
    coords.set_item("range", (("range",), range_arr))?;
    if let Some(lat) = meta.latitude {
        coords.set_item("latitude", lat)?;
    }
    if let Some(lon) = meta.longitude {
        coords.set_item("longitude", lon)?;
    }
    if let Some(alt) = meta.altitude {
        coords.set_item("altitude", alt)?;
    }

    // Process each moment type - pad all to max_n_gates
    for (moment_getter, cf_name) in &MOMENT_NAMES {
        if moment_info.contains_key(*cf_name) {
            // Create 2D data array (time, range) filled with NaN, padded to max_n_gates
            let mut moment_data: Vec<f32> = vec![f32::NAN; n_rays * max_n_gates];

            for (out_idx, &radial_idx) in sorted_indices.iter().enumerate() {
                let radial = &radials[radial_idx];
                if let Some(moment) = get_moment_data(radial, moment_getter) {
                    let values = moment.values();
                    for (gate_idx, value) in values.iter().enumerate() {
                        if gate_idx >= max_n_gates {
                            break;
                        }
                        let flat_idx = out_idx * max_n_gates + gate_idx;
                        moment_data[flat_idx] = match value {
                            MomentValue::Value(v) => *v,
                            MomentValue::BelowThreshold => f32::NAN,
                            MomentValue::RangeFolded => f32::NAN,
                        };
                    }
                }
            }

            // Create numpy array and reshape to 2D
            let flat_arr = moment_data.into_pyarray(py);
            let arr_2d = flat_arr.call_method1("reshape", ((n_rays, max_n_gates),))?;

            // Add to data_vars with dimensions
            let dims = ("time", "range");
            data_vars.set_item(*cf_name, (dims, arr_2d))?;
        }
    }

    // Add sweep-level scalar variables (xradar-compatible)
    data_vars.set_item("sweep_number", sweep_idx)?;
    if let Some(first_radial) = radials.first() {
        data_vars.set_item("sweep_fixed_angle", first_radial.elevation_angle_degrees())?;
    }

    // Create Dataset
    let kwargs = [
        ("data_vars", data_vars.as_any()),
        ("coords", coords.as_any()),
    ]
    .into_py_dict(py)?;
    let dataset = xr.call_method("Dataset", (), Some(&kwargs))?;

    Ok(dataset)
}

/// Helper to get moment data from a radial by name
fn get_moment_data<'a>(
    radial: &'a Radial,
    moment_name: &str,
) -> Option<&'a nexrad_model::data::MomentData> {
    match moment_name {
        "reflectivity" => radial.reflectivity(),
        "velocity" => radial.velocity(),
        "spectrum_width" => radial.spectrum_width(),
        "differential_reflectivity" => radial.differential_reflectivity(),
        "differential_phase" => radial.differential_phase(),
        "correlation_coefficient" => radial.correlation_coefficient(),
        "specific_differential_phase" => radial.specific_differential_phase(),
        _ => None,
    }
}
