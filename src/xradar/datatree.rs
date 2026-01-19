//! DataTree conversion for xradar compatibility

use crate::error::{RadrsError, Result};
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{MomentValue, Radial, Scan, Sweep};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
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
#[pyo3(name = "open_datatree")]
pub fn open_datatree_py(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    // Handle different input types
    let data = if source.is_instance_of::<PyBytes>() {
        // Bytes input
        source.extract::<Vec<u8>>()?
    } else {
        // String path or URL
        let path_str: String = source.extract()?;

        if path_str.starts_with("s3://") {
            // S3 URL - use async fetch
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(fetch_s3_file(&path_str))?
        } else {
            // Local file path
            fs::read(&path_str).map_err(RadrsError::Io)?
        }
    };

    // Parse NEXRAD data
    let scan = parse_nexrad_data(&data)?;

    // Convert to DataTree
    scan_to_datatree(py, &scan)
}

/// Open a NEXRAD file from path, URL, or bytes
pub fn open_datatree(source: &[u8]) -> Result<Scan> {
    parse_nexrad_data(source)
}

/// Parse raw NEXRAD data into a Scan
fn parse_nexrad_data(data: &[u8]) -> Result<Scan> {
    let volume = VolumeFile::new(data.to_vec());
    let scan = volume.scan()?;
    Ok(scan)
}

/// Fetch a file from S3
async fn fetch_s3_file(url: &str) -> Result<Vec<u8>> {
    use object_store::aws::AmazonS3Builder;
    use object_store::path::Path as ObjectPath;
    use object_store::ObjectStore;

    // Parse S3 URL: s3://bucket/path/to/file
    let url = url.strip_prefix("s3://").ok_or_else(|| {
        RadrsError::InvalidUrl(format!("Invalid S3 URL: {}", url))
    })?;

    let (bucket, key) = url.split_once('/').ok_or_else(|| {
        RadrsError::InvalidUrl(format!("Invalid S3 URL format: s3://{}", url))
    })?;

    // Build S3 client with anonymous access (NEXRAD data is public)
    let store = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .build()?;

    let path = ObjectPath::from(key);
    let result = store.get(&path).await?;
    let bytes = result.bytes().await?;

    Ok(bytes.to_vec())
}

/// Convert a Scan to an xarray DataTree
fn scan_to_datatree(py: Python<'_>, scan: &Scan) -> PyResult<PyObject> {
    // Import xarray
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    // Build root attributes
    let root_attrs = PyDict::new(py);
    root_attrs.set_item("Conventions", "CF-1.8")?;
    root_attrs.set_item("instrument_type", "radar")?;
    root_attrs.set_item("platform_type", "fixed")?;
    root_attrs.set_item("volume_coverage_pattern", scan.coverage_pattern_number())?;

    // Build the DataTree structure - create root with attrs
    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;

    // Create DataTree - xarray 2024+ uses from_dict
    let tree_dict = PyDict::new(py);
    tree_dict.set_item("/", root_ds)?;

    for (sweep_idx, sweep) in scan.sweeps().iter().enumerate() {
        let sweep_name = format!("/sweep_{}", sweep_idx);
        let dataset = sweep_to_dataset(py, &xr, &np, sweep)?;
        tree_dict.set_item(&sweep_name, dataset)?;
    }

    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.into())
}

/// Convert a Sweep to an xarray Dataset
fn sweep_to_dataset<'py>(
    py: Python<'py>,
    xr: &Bound<'py, PyModule>,
    np: &Bound<'py, PyModule>,
    sweep: &Sweep,
) -> PyResult<Bound<'py, PyAny>> {
    let radials = sweep.radials();
    if radials.is_empty() {
        // Return empty dataset
        return xr.call_method1("Dataset", (PyDict::new(py),));
    }

    let n_rays = radials.len();

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

                let entry = moment_info.entry(*cf_name).or_insert((0, first_range, gate_interval));
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

    // Build coordinate arrays
    let mut azimuth_data: Vec<f32> = Vec::with_capacity(n_rays);
    let mut elevation_data: Vec<f32> = Vec::with_capacity(n_rays);
    let mut time_data: Vec<i64> = Vec::with_capacity(n_rays);

    for radial in radials {
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

    // Process each moment type - pad all to max_n_gates
    for (moment_getter, cf_name) in &MOMENT_NAMES {
        if moment_info.contains_key(*cf_name) {
            // Create 2D data array (time, range) filled with NaN, padded to max_n_gates
            let mut moment_data: Vec<f32> = vec![f32::NAN; n_rays * max_n_gates];

            for (ray_idx, radial) in radials.iter().enumerate() {
                if let Some(moment) = get_moment_data(radial, moment_getter) {
                    let values = moment.values();
                    for (gate_idx, value) in values.iter().enumerate() {
                        if gate_idx >= max_n_gates {
                            break;
                        }
                        let flat_idx = ray_idx * max_n_gates + gate_idx;
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

    // Create Dataset
    let kwargs = [("data_vars", data_vars.as_any()), ("coords", coords.as_any())].into_py_dict(py)?;
    let dataset = xr.call_method("Dataset", (), Some(&kwargs))?;

    // Add sweep-level attributes
    let sweep_attrs = PyDict::new(py);
    sweep_attrs.set_item("sweep_number", sweep.elevation_number())?;
    if let Some(first_radial) = radials.first() {
        sweep_attrs.set_item("sweep_fixed_angle", first_radial.elevation_angle_degrees())?;
    }
    dataset.setattr("attrs", sweep_attrs)?;

    Ok(dataset)
}

/// Helper to get moment data from a radial by name
fn get_moment_data<'a>(radial: &'a Radial, moment_name: &str) -> Option<&'a nexrad_model::data::MomentData> {
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
