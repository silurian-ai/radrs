//! DataTree conversion for xradar compatibility

use crate::constants::XRADAR_MOMENT_NAMES;
use crate::error::{RadrsError, Result};
use crate::fetch::{RUNTIME, default_open_datatree_storage_options, fetch_bytes_from_url};
use crate::metadata::{ScanMeta, extract_scan_meta};
use crate::metadata_build::{build_root_attrs, build_root_vars, set_sweep_mode_scalars};
use crate::range::{
    RangeGeometry, canonical_lattice, gate_center_m, geometry, map_gate_to_lattice,
};
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::{
    CFPMomentData, CFPMomentValue, ElevationCut, MomentData, MomentValue, Radial, Scan, Sweep,
};
use numpy::IntoPyArray;
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::time::Instant;

/// Open a NEXRAD Level 2 file and return an xarray DataTree
///
/// # Arguments
/// * `source` - Bytes, a local path, or a URI (`s3://`, `gs://`, `az://`,
///   `file://`). The URI must point at a single NEXRAD Level 2 volume.
///   Tar archives — including the public GCS NEXRAD mirror's 6-minute
///   bundles at `gs://gcp-public-data-nexrad-l2` — are not unpacked here
///   and will fail at parse. For the public archive, S3
///   (`s3://unidata-nexrad-level2`) is the only out-of-the-box source
///   today; `az://` is plumbed but no public NEXRAD Azure mirror is
///   known at the time of writing.
/// * `storage_options` - Optional storage credentials forwarded to
///   object_store (e.g. `{"anon": "true"}`)
///
/// # Returns
/// xarray.DataTree with sweeps as children
#[pyfunction]
#[pyo3(name = "open_datatree", signature = (source, sort_by_azimuth = false, storage_options = None))]
pub fn open_datatree_py(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    sort_by_azimuth: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    // Handle different input types
    let data = if source.is_instance_of::<PyBytes>() {
        // Bytes input
        source.extract::<Vec<u8>>()?
    } else {
        // String path or URL — supports s3://, gs://, az://, and local
        let path_str: String = source.extract()?;
        let storage_options =
            storage_options.or_else(|| default_open_datatree_storage_options(&path_str));
        RUNTIME
            .block_on(fetch_bytes_from_url(&path_str, storage_options))?
            .to_vec()
    };

    // Parse NEXRAD data without holding the GIL
    let (scan, meta) = py.detach(|| parse_nexrad_data(data))?;

    // Convert to DataTree
    scan_to_datatree(py, &scan, &meta, sort_by_azimuth)
}

/// Open a NEXRAD Level 2 file asynchronously and return an xarray DataTree.
#[pyfunction]
#[pyo3(name = "open_datatree_async", signature = (source, sort_by_azimuth = false, storage_options = None))]
pub fn open_datatree_async_py(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    sort_by_azimuth: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let source = source.as_borrowed().to_owned().unbind();

    let awaitable = future_into_py(py, async move {
        let data = fetch_source_bytes_async(source, storage_options).await?;

        let (scan, meta) = tokio::task::spawn_blocking(move || parse_nexrad_data(data))
            .await
            .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;

        Python::attach(|py| scan_to_datatree(py, &scan, &meta, sort_by_azimuth))
    })?;

    Ok(awaitable.into())
}

/// Open a NEXRAD file from path, URL, or bytes
pub fn open_datatree(source: Vec<u8>) -> Result<(Scan, ScanMeta)> {
    parse_nexrad_data(source)
}

/// Parse raw NEXRAD data into a Scan
fn parse_nexrad_data(data: Vec<u8>) -> Result<(Scan, ScanMeta)> {
    let bytes_len = data.len();
    let start = Instant::now();
    let volume = VolumeFile::new(data);
    let meta = extract_scan_meta(&volume);
    let scan = volume.scan().map_err(|err| {
        tracing::warn!(
            target: "radrs::parse",
            format = "xradar",
            bytes = bytes_len,
            error = %err
        );
        err
    })?;
    let elapsed = start.elapsed();
    tracing::info!(
        target: "radrs::parse",
        format = "xradar",
        bytes = bytes_len,
        vcp = u16::from(scan.coverage_pattern_number()),
        n_sweeps = scan.sweeps().len(),
        elapsed_ms = elapsed.as_millis() as u64
    );
    Ok((scan, meta))
}

async fn fetch_source_bytes_async(
    source: Py<PyAny>,
    storage_options: Option<HashMap<String, String>>,
) -> Result<Vec<u8>> {
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
            let storage_options =
                storage_options.or_else(|| default_open_datatree_storage_options(&path));
            fetch_bytes_from_url(&path, storage_options)
                .await
                .map(|b| b.to_vec())
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

    // Compute time coverage from all radials
    let (time_start, time_end) = compute_time_coverage(scan);

    // Build root attributes
    let root_attrs = build_root_attrs(
        py,
        scan.coverage_pattern_number().into(),
        meta.instrument_name.as_deref(),
    )?;

    // Root data variables (xradar-style metadata)
    let root_vars = build_root_vars(py, meta, time_start, time_end)?;

    // Build the DataTree structure - create root with attrs and vars
    let root_kwargs = [("data_vars", root_vars.as_any())].into_py_dict(py)?;
    let root_ds = xr.call_method("Dataset", (), Some(&root_kwargs))?;
    root_ds.setattr("attrs", root_attrs)?;

    // Create DataTree - xarray 2024+ uses from_dict
    let tree_dict = PyDict::new(py);
    tree_dict.set_item("/", root_ds)?;

    // Nominal sweep angles live in the VCP's elevation-cut table, not in the radials
    let elevation_cuts = scan.coverage_pattern().elevation_cuts();

    // Add sweep datasets
    for (sweep_idx, sweep) in scan.sweeps().iter().enumerate() {
        let sweep_name = format!("/sweep_{}", sweep_idx);
        let dataset = sweep_to_dataset(
            py,
            &xr,
            &np,
            sweep,
            sweep_idx,
            meta,
            sort_by_azimuth,
            elevation_cuts,
        )?;
        tree_dict.set_item(&sweep_name, dataset)?;
    }

    // Add CF-radial metadata groups (empty datasets with coordinates, for compatibility)
    let metadata_ds = create_metadata_dataset(py, &xr, meta)?;
    tree_dict.set_item("/radar_parameters", &metadata_ds)?;
    tree_dict.set_item("/georeferencing_correction", &metadata_ds)?;
    // radar_calibration has no coordinates in xradar
    let empty_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    tree_dict.set_item("/radar_calibration", empty_ds)?;

    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.unbind())
}

/// Compute min/max timestamps across all sweeps
fn compute_time_coverage(scan: &Scan) -> (Option<i64>, Option<i64>) {
    let mut min_ts: Option<i64> = None;
    let mut max_ts: Option<i64> = None;

    for sweep in scan.sweeps() {
        for radial in sweep.radials() {
            let ts = radial.collection_timestamp();
            min_ts = Some(min_ts.map_or(ts, |m| m.min(ts)));
            max_ts = Some(max_ts.map_or(ts, |m| m.max(ts)));
        }
    }

    (min_ts, max_ts)
}

/// Create a metadata dataset with just coordinates (for radar_parameters, georeferencing_correction)
fn create_metadata_dataset<'py>(
    py: Python<'py>,
    xr: &Bound<'py, PyModule>,
    meta: &ScanMeta,
) -> PyResult<Bound<'py, PyAny>> {
    let coords = PyDict::new(py);
    if let Some(lat) = meta.latitude {
        coords.set_item("latitude", lat)?;
    }
    if let Some(lon) = meta.longitude {
        coords.set_item("longitude", lon)?;
    }
    if let Some(alt) = meta.altitude {
        coords.set_item("altitude", alt)?;
    }

    let kwargs = [("coords", coords.as_any())].into_py_dict(py)?;
    xr.call_method("Dataset", (), Some(&kwargs))
}

/// Convert a Sweep to an xarray Dataset
#[allow(clippy::too_many_arguments)]
fn sweep_to_dataset<'py>(
    py: Python<'py>,
    xr: &Bound<'py, PyModule>,
    np: &Bound<'py, PyModule>,
    sweep: &Sweep,
    sweep_idx: usize,
    meta: &ScanMeta,
    sort_by_azimuth: bool,
    elevation_cuts: &[ElevationCut],
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

    // Build one exact physical lattice containing every moment's gate centers.
    // Values are placed by physical gate center rather than by source index.
    let mut moment_info: HashMap<&str, ()> = HashMap::new();
    let mut geometries = Vec::<RangeGeometry>::new();

    for radial in radials {
        for (moment_getter, cf_name) in &XRADAR_MOMENT_NAMES {
            if let Some(moment) = get_moment_data(radial, moment_getter) {
                moment_info.insert(*cf_name, ());
                geometries.push(moment.geometry()?);
            }
        }
    }

    // If no moment data, return empty dataset
    if moment_info.is_empty() {
        return xr.call_method1("Dataset", (PyDict::new(py),));
    }

    let range_grid = canonical_lattice(&geometries)?
        .ok_or_else(|| RadrsError::MissingData("sweep has no moment data".into()))?;
    let range_data_m: Vec<u32> = (0..range_grid.gate_count)
        .map(|gate| gate_center_m(range_grid, gate))
        .collect::<Result<Vec<_>>>()?;
    let n_range = range_grid.gate_count;

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

    // Use azimuth as the primary dimension (xradar convention)
    let ray_dim = "azimuth";

    // Create azimuth coordinate
    let azimuth_arr = azimuth_data.into_pyarray(py);
    coords.set_item("azimuth", ((ray_dim,), azimuth_arr))?;

    // Create elevation coordinate
    let elevation_arr = elevation_data.into_pyarray(py);
    coords.set_item("elevation", ((ray_dim,), elevation_arr))?;

    // Create time coordinate (as int64 nanoseconds for datetime64[ns])
    // Convert milliseconds to datetime64[ns]
    let time_arr_i64 = time_data.into_pyarray(py);
    let time_arr = np.call_method1("array", (time_arr_i64,))?;
    let time_arr_ms = time_arr.call_method1("astype", ("datetime64[ms]",))?;
    let time_arr_ns = time_arr_ms.call_method1("astype", ("datetime64[ns]",))?;
    coords.set_item("time", ((ray_dim,), time_arr_ns))?;

    // Build range coordinate from physical gate centers (in metres).
    let range_data: Vec<f64> = range_data_m.iter().map(|range| *range as f64).collect();
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
    for (moment_getter, cf_name) in &XRADAR_MOMENT_NAMES {
        if moment_info.contains_key(*cf_name) {
            // Create 2D data array (time, range) filled with NaN, padded to max_n_gates
            let mut moment_data: Vec<f32> = vec![f32::NAN; n_rays * n_range];

            for (out_idx, &radial_idx) in sorted_indices.iter().enumerate() {
                let radial = &radials[radial_idx];
                if let Some(moment) = get_moment_data(radial, moment_getter) {
                    let moment_geometry = moment.geometry()?;
                    let values = moment.values_f32();
                    for (gate_idx, value) in values.iter().enumerate() {
                        let range_idx = map_gate_to_lattice(moment_geometry, gate_idx, range_grid)?;
                        let flat_idx = out_idx * n_range + range_idx;
                        moment_data[flat_idx] = *value;
                    }
                }
            }

            // Create numpy array and reshape to 2D
            let flat_arr = moment_data.into_pyarray(py);
            let arr_2d = flat_arr.call_method1("reshape", ((n_rays, n_range),))?;

            // Add to data_vars with dimensions
            let dims = (ray_dim, "range");
            data_vars.set_item(*cf_name, (dims, arr_2d))?;
        }
    }

    // Add sweep-level scalar variables (xradar-compatible)
    data_vars.set_item("sweep_number", sweep_idx)?;
    // `sweep_fixed_angle` is the sweep's *nominal* target angle (CF/FM301).  Falls back to the first radial's
    // measured elevation when the cut table does not cover this elevation number.
    let elevation_number = sweep.elevation_number() as usize;
    let nominal_angle = elevation_cuts
        .get(elevation_number.wrapping_sub(1))
        .map(|cut| cut.elevation_angle_degrees() as f32);
    if let Some(angle) =
        nominal_angle.or_else(|| radials.first().map(|r| r.elevation_angle_degrees()))
    {
        data_vars.set_item("sweep_fixed_angle", angle)?;
    }
    set_sweep_mode_scalars(&data_vars)?;

    // Create Dataset
    let kwargs = [
        ("data_vars", data_vars.as_any()),
        ("coords", coords.as_any()),
    ]
    .into_py_dict(py)?;
    let dataset = xr.call_method("Dataset", (), Some(&kwargs))?;

    Ok(dataset)
}

enum MomentRef<'a> {
    Standard(&'a MomentData),
    Cfp(&'a CFPMomentData),
}

impl MomentRef<'_> {
    fn geometry(&self) -> Result<RangeGeometry> {
        match self {
            Self::Standard(moment) => geometry(*moment),
            Self::Cfp(moment) => geometry(*moment),
        }
    }

    fn values_f32(&self) -> Vec<f32> {
        match self {
            Self::Standard(moment) => moment
                .values()
                .into_iter()
                .map(|value| match value {
                    MomentValue::Value(v) => v,
                    MomentValue::BelowThreshold | MomentValue::RangeFolded => f32::NAN,
                })
                .collect(),
            Self::Cfp(moment) => moment
                .values()
                .into_iter()
                .map(|value| match value {
                    CFPMomentValue::Value(v) => v,
                    CFPMomentValue::Status(_) => f32::NAN,
                })
                .collect(),
        }
    }
}

/// Helper to get moment data from a radial by name
fn get_moment_data<'a>(radial: &'a Radial, moment_name: &str) -> Option<MomentRef<'a>> {
    match moment_name {
        name if name == XRADAR_MOMENT_NAMES[0].0 => radial.reflectivity().map(MomentRef::Standard),
        name if name == XRADAR_MOMENT_NAMES[1].0 => radial.velocity().map(MomentRef::Standard),
        name if name == XRADAR_MOMENT_NAMES[2].0 => {
            radial.spectrum_width().map(MomentRef::Standard)
        }
        name if name == XRADAR_MOMENT_NAMES[3].0 => {
            radial.differential_reflectivity().map(MomentRef::Standard)
        }
        name if name == XRADAR_MOMENT_NAMES[4].0 => {
            radial.differential_phase().map(MomentRef::Standard)
        }
        name if name == XRADAR_MOMENT_NAMES[5].0 => {
            radial.correlation_coefficient().map(MomentRef::Standard)
        }
        name if name == XRADAR_MOMENT_NAMES[6].0 => {
            radial.clutter_filter_power().map(MomentRef::Cfp)
        }
        _ => None,
    }
}
