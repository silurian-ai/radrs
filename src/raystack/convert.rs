//! Conversion between DataTree and Raystack formats

use crate::raystack::fold::fold_ranges_into;
use crate::raystack::parse::{RaystackData, SweepInfo, DEFAULT_FOLD_SIZE, raystack_to_python};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

/// Convert xarray DataTree to Raystack format
#[pyfunction]
#[pyo3(name = "from_datatree", signature = (datatree, fold_size = None))]
pub fn from_datatree_py(
    py: Python<'_>,
    datatree: &Bound<'_, PyAny>,
    fold_size: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let raystack = datatree_to_raystack(py, datatree, fold_size)?;
    raystack_to_python(py, raystack, None)
}

/// Convert Raystack to xarray DataTree
#[pyfunction]
#[pyo3(name = "to_datatree")]
pub fn to_datatree_py(py: Python<'_>, raystack_dict: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
    raystack_dict_to_datatree(py, raystack_dict)
}

/// Convert Raystack dict to raystack-style DataTree (vcps/sweeps/returns)
#[pyfunction]
#[pyo3(name = "to_raystack_datatree")]
pub fn to_raystack_datatree_py(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    raystack_dict_to_raystack_datatree(py, raystack_dict)
}

/// Extract sweep number from sweep name like "sweep_0" -> Some(0)
fn extract_sweep_number(name: &str) -> Option<usize> {
    name.strip_prefix("sweep_").and_then(|n| n.parse().ok())
}

/// First pass: count radials per sweep for preallocation
fn count_radials(
    py: Python<'_>,
    children_dict: &Bound<'_, PyDict>,
) -> PyResult<(Vec<(String, usize, u8, f32)>, usize)> {
    let np = py.import("numpy")?;

    // Collect and sort sweep names numerically
    let mut sweep_items: Vec<_> = children_dict.iter().collect();
    sweep_items.sort_by(|(k1, _), (k2, _)| {
        let s1: String = k1.extract().unwrap_or_default();
        let s2: String = k2.extract().unwrap_or_default();
        // Sort numerically: sweep_0, sweep_1, ..., sweep_10, sweep_11, etc.
        match (extract_sweep_number(&s1), extract_sweep_number(&s2)) {
            (Some(n1), Some(n2)) => n1.cmp(&n2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => s1.cmp(&s2),
        }
    });

    let mut sweep_info = Vec::new();
    let mut total_radials = 0;
    let mut sweep_counter: u8 = 0;

    for (key, child) in sweep_items.iter() {
        let sweep_name: String = key.extract()?;
        if !sweep_name.starts_with("sweep_") {
            continue;
        }

        let dataset = child.getattr("dataset")?;

        // Get azimuth to count radials
        let n_radials = if let Ok(azimuth) = dataset.get_item("azimuth") {
            let values = azimuth.getattr("values")?;
            let flat = np.call_method1("ravel", (&values,))?;
            let data: Vec<f32> = flat.extract()?;
            data.len()
        } else {
            continue;
        };

        // Get sweep metadata - use sweep_counter for fallback, not enumeration index
        let sweep_attrs = dataset.getattr("attrs")?;
        let elevation_number: u8 = sweep_attrs
            .get_item("sweep_number")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(sweep_counter);

        // Get elevation angle from first radial
        let elevation_angle: f32 = if let Ok(elevation) = dataset.get_item("elevation") {
            let values = elevation.getattr("values")?;
            let flat = np.call_method1("ravel", (&values,))?;
            let data: Vec<f32> = flat.extract()?;
            data.first().copied().unwrap_or(0.0)
        } else {
            sweep_attrs
                .get_item("sweep_fixed_angle")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0)
        };

        sweep_info.push((sweep_name, n_radials, elevation_number, elevation_angle));
        total_radials += n_radials;
        sweep_counter += 1;
    }

    Ok((sweep_info, total_radials))
}

/// Convert DataTree to internal Raystack representation with preallocation
fn datatree_to_raystack(
    py: Python<'_>,
    datatree: &Bound<'_, PyAny>,
    fold_size: usize,
) -> PyResult<RaystackData> {
    let np = py.import("numpy")?;

    // Get VCP from root attributes
    let root_attrs = datatree.getattr("attrs")?;
    let pattern_number: u16 = root_attrs
        .get_item("volume_coverage_pattern")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);

    // Get children (sweeps) - convert Frozen to dict
    let children = datatree.getattr("children")?;
    let builtins = py.import("builtins")?;
    let children_dict = builtins.call_method1("dict", (&children,))?;
    let children_dict = children_dict.cast::<PyDict>()?;

    // First pass: count radials for preallocation
    let (sweep_meta, total_radials) = count_radials(py, children_dict)?;

    if total_radials == 0 {
        return Ok(RaystackData {
            pattern_number,
            sweeps: Vec::new(),
            n_radials: 0,
            fold_size,
            azimuth: Vec::new(),
            elevation: Vec::new(),
            time: Vec::new(),
            sweep_idx: Vec::new(),
            dbzh: Vec::new(),
            vradh: Vec::new(),
            wradh: Vec::new(),
            zdr: Vec::new(),
            phidp: Vec::new(),
            rhohv: Vec::new(),
            kdp: Vec::new(),
        });
    }

    // Preallocate
    let moment_len = total_radials * fold_size;
    let mut raystack = RaystackData {
        pattern_number,
        sweeps: Vec::with_capacity(sweep_meta.len()),
        n_radials: total_radials,
        fold_size,
        azimuth: vec![0.0; total_radials],
        elevation: vec![0.0; total_radials],
        time: vec![0; total_radials],
        sweep_idx: vec![0; total_radials],
        dbzh: vec![f32::NAN; moment_len],
        vradh: vec![f32::NAN; moment_len],
        wradh: vec![f32::NAN; moment_len],
        zdr: vec![f32::NAN; moment_len],
        phidp: vec![f32::NAN; moment_len],
        rhohv: vec![f32::NAN; moment_len],
        kdp: vec![f32::NAN; moment_len],
    };

    // Build sweep info
    let mut start_index = 0;
    for (_, n_radials, elevation_number, elevation_angle) in &sweep_meta {
        raystack.sweeps.push(SweepInfo {
            elevation_number: *elevation_number,
            elevation_angle: *elevation_angle,
            n_radials: *n_radials,
            start_index,
        });
        start_index += n_radials;
    }

    // Collect and sort sweep items again for second pass (numerically)
    let mut sweep_items: Vec<_> = children_dict.iter().collect();
    sweep_items.sort_by(|(k1, _), (k2, _)| {
        let s1: String = k1.extract().unwrap_or_default();
        let s2: String = k2.extract().unwrap_or_default();
        // Sort numerically: sweep_0, sweep_1, ..., sweep_10, sweep_11, etc.
        match (extract_sweep_number(&s1), extract_sweep_number(&s2)) {
            (Some(n1), Some(n2)) => n1.cmp(&n2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => s1.cmp(&s2),
        }
    });

    // Second pass: fill data
    // Use separate sweep counter that only counts actual sweeps, not all children
    let mut radial_idx = 0;
    let mut sweep_counter: u32 = 0;
    for (key, child) in sweep_items.iter() {
        let sweep_name: String = key.extract()?;
        if !sweep_name.starts_with("sweep_") {
            continue;
        }

        let dataset = child.getattr("dataset")?;

        // Get coordinates
        let azimuth_data: Vec<f32> = if let Ok(azimuth) = dataset.get_item("azimuth") {
            let values = azimuth.getattr("values")?;
            let flat = np.call_method1("ravel", (&values,))?;
            flat.extract()?
        } else {
            continue;
        };

        let elevation_data: Vec<f32> = if let Ok(elevation) = dataset.get_item("elevation") {
            let values = elevation.getattr("values")?;
            let flat = np.call_method1("ravel", (&values,))?;
            flat.extract()?
        } else {
            vec![0.0; azimuth_data.len()]
        };

        let time_data: Vec<i64> = if let Ok(time) = dataset.get_item("time") {
            let values = time.getattr("values")?;
            let as_int = values.call_method1("astype", ("int64",))?;
            let flat = np.call_method1("ravel", (&as_int,))?;
            let ms: Vec<i64> = flat.extract()?;
            ms.iter().map(|x| x / 1_000_000).collect()
        } else {
            vec![0; azimuth_data.len()]
        };

        let n_radials = azimuth_data.len();

        // Fill coordinate arrays - use sweep_counter which only counts actual sweeps
        for i in 0..n_radials {
            raystack.azimuth[radial_idx + i] = azimuth_data[i];
            raystack.elevation[radial_idx + i] = elevation_data[i];
            raystack.time[radial_idx + i] = time_data[i];
            raystack.sweep_idx[radial_idx + i] = sweep_counter;
        }

        // Extract and fold moment data
        let moment_names = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"];

        for (moment_idx, moment_name) in moment_names.iter().enumerate() {
            if let Ok(moment_var) = dataset.get_item(*moment_name) {
                let values = moment_var.getattr("values")?;
                let arr: Vec<Vec<f32>> = values.extract()?;

                for (row_idx, row) in arr.iter().enumerate() {
                    let dest_start = (radial_idx + row_idx) * fold_size;
                    let dest = match moment_idx {
                        0 => &mut raystack.dbzh[dest_start..dest_start + fold_size],
                        1 => &mut raystack.vradh[dest_start..dest_start + fold_size],
                        2 => &mut raystack.wradh[dest_start..dest_start + fold_size],
                        3 => &mut raystack.zdr[dest_start..dest_start + fold_size],
                        4 => &mut raystack.phidp[dest_start..dest_start + fold_size],
                        5 => &mut raystack.rhohv[dest_start..dest_start + fold_size],
                        6 => &mut raystack.kdp[dest_start..dest_start + fold_size],
                        _ => unreachable!(),
                    };
                    fold_ranges_into(row, dest);
                }
            }
            // Missing moments stay as NaN (already initialized)
        }

        radial_idx += n_radials;
        sweep_counter += 1;
    }

    Ok(raystack)
}

/// Convert Raystack dict back to xarray DataTree
fn raystack_dict_to_datatree(py: Python<'_>, raystack_dict: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    // Extract data from dict
    let vcps = raystack_dict.get_item("vcps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'vcps' in raystack dict")
    })?;
    let sweeps = raystack_dict.get_item("sweeps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'sweeps' in raystack dict")
    })?;
    let returns = raystack_dict.get_item("returns")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'returns' in raystack dict")
    })?;

    let pattern_number: u16 = vcps.get_item("pattern_number")?.extract()?;
    let sweeps_list: Vec<Bound<'_, PyAny>> = sweeps.extract()?;

    // Create root attributes
    let root_attrs = PyDict::new(py);
    root_attrs.set_item("Conventions", "CF-1.8")?;
    root_attrs.set_item("instrument_type", "radar")?;
    root_attrs.set_item("volume_coverage_pattern", pattern_number)?;

    // Build tree dict
    let tree_dict = PyDict::new(py);

    // Root dataset
    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;
    tree_dict.set_item("/", root_ds)?;

    // Get global arrays
    let azimuth_arr = returns.get_item("azimuth")?;
    let elevation_arr = returns.get_item("elevation")?;
    let time_arr = returns.get_item("time")?;

    // Get moment names from returns
    let moment_names = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"];

    // Process each sweep
    for (sweep_idx, sweep_info) in sweeps_list.iter().enumerate() {
        let start_index: usize = sweep_info.get_item("start_index")?.extract()?;
        let n_radials: usize = sweep_info.get_item("n_radials")?.extract()?;
        let elevation_number: u8 = sweep_info.get_item("elevation_number")?.extract()?;
        let elevation_angle: f32 = sweep_info.get_item("elevation_angle")?.extract()?;

        let end_index = start_index + n_radials;

        // Slice arrays for this sweep using Python slice objects
        let slice_obj = py.import("builtins")?.getattr("slice")?.call1((start_index, end_index))?;
        let sweep_azimuth = azimuth_arr.get_item(&slice_obj)?;
        let sweep_elevation = elevation_arr.get_item(&slice_obj)?;
        let sweep_time = time_arr.get_item(&slice_obj)?;

        // Convert time to datetime64
        let sweep_time_arr = np.call_method1("array", (&sweep_time,))?;
        let sweep_time_dt = sweep_time_arr.call_method1("astype", ("datetime64[ms]",))?;

        // Build coords
        let coords = PyDict::new(py);
        coords.set_item("azimuth", (("time",), sweep_azimuth))?;
        coords.set_item("elevation", (("time",), sweep_elevation))?;
        coords.set_item("time", (("time",), sweep_time_dt))?;

        // Build data vars
        let data_vars = PyDict::new(py);
        let mut range_added = false;

        for moment_name in &moment_names {
            if let Ok(moment_arr) = returns.get_item(*moment_name) {
                let sweep_moment = moment_arr.get_item(&slice_obj)?;

                // Get range dimension size
                let shape = sweep_moment.getattr("shape")?;
                let shape_tuple: (usize, usize) = shape.extract()?;
                let n_range = shape_tuple.1;

                // Create range coordinate if not exists
                if !range_added {
                    // Default range values (would need actual metadata for accuracy)
                    let range_data: Vec<f64> = (0..n_range)
                        .map(|i| i as f64 * 250.0)
                        .collect();
                    let range_arr = range_data.into_pyarray(py);
                    coords.set_item("range", (("range",), range_arr))?;
                    range_added = true;
                }

                data_vars.set_item(*moment_name, (("time", "range"), sweep_moment))?;
            }
        }

        // Create dataset
        let kwargs = [("data_vars", data_vars.as_any()), ("coords", coords.as_any())].into_py_dict(py)?;
        let dataset = xr.call_method("Dataset", (), Some(&kwargs))?;

        // Set sweep attributes
        let sweep_attrs = PyDict::new(py);
        sweep_attrs.set_item("sweep_number", elevation_number)?;
        sweep_attrs.set_item("sweep_fixed_angle", elevation_angle)?;
        dataset.setattr("attrs", sweep_attrs)?;

        let sweep_name = format!("/sweep_{}", sweep_idx);
        tree_dict.set_item(&sweep_name, dataset)?;
    }

    // Create DataTree
    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.into())
}

/// Convert Raystack dict to a flat raystack DataTree with vcps/sweeps/returns datasets
fn raystack_dict_to_raystack_datatree(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    let vcps = raystack_dict.get_item("vcps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'vcps' in raystack dict")
    })?;
    let sweeps = raystack_dict.get_item("sweeps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'sweeps' in raystack dict")
    })?;
    let returns = raystack_dict.get_item("returns")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'returns' in raystack dict")
    })?;

    let pattern_number: u16 = vcps.get_item("pattern_number")?.extract()?;
    let sweeps_list: Vec<Bound<'_, PyAny>> = sweeps.extract()?;

    // Extract time array and derive sweep/return time coordinates
    let time_arr = returns.get_item("time")?;
    let time_vec: Vec<i64> = time_arr.extract().unwrap_or_default();
    let n_returns = time_vec.len();

    let vcp_time = time_vec.iter().copied().min().unwrap_or(0);

    let mut sweep_times: Vec<i64> = Vec::with_capacity(sweeps_list.len());
    let mut sweep_time_per_return = vec![vcp_time; n_returns];

    for sweep_info in sweeps_list.iter() {
        let start_index: usize = sweep_info.get_item("start_index")?.extract()?;
        let n_radials: usize = sweep_info.get_item("n_radials")?.extract()?;
        let sweep_time = time_vec.get(start_index).copied().unwrap_or(vcp_time);
        sweep_times.push(sweep_time);
        let end_index = start_index.saturating_add(n_radials).min(n_returns);
        for idx in start_index..end_index {
            sweep_time_per_return[idx] = sweep_time;
        }
    }

    // Build datetime64[ms] arrays
    let vcp_time_arr = np.call_method1("array", (&vec![vcp_time],))?;
    let vcp_time_dt = vcp_time_arr.call_method1("astype", ("datetime64[ms]",))?;

    let sweep_time_arr = np.call_method1("array", (&sweep_times,))?;
    let sweep_time_dt = sweep_time_arr.call_method1("astype", ("datetime64[ms]",))?;

    let return_time_arr = np.call_method1("array", (&time_vec,))?;
    let return_time_dt = return_time_arr.call_method1("astype", ("datetime64[ms]",))?;

    let sweep_time_per_return_arr = np.call_method1("array", (&sweep_time_per_return,))?;
    let sweep_time_per_return_dt = sweep_time_per_return_arr.call_method1("astype", ("datetime64[ms]",))?;

    // Build vcps dataset
    let vcps_coords = PyDict::new(py);
    vcps_coords.set_item("vcp_time", (("vcp_time",), vcp_time_dt))?;

    let vcps_vars = PyDict::new(py);
    vcps_vars.set_item("pattern_number", (("vcp_time",), vec![pattern_number]))?;

    let vcps_ds = xr.call_method(
        "Dataset",
        (),
        Some(&[("data_vars", vcps_vars.as_any()), ("coords", vcps_coords.as_any())].into_py_dict(py)?),
    )?;

    // Build sweeps dataset
    let mut elevation_numbers: Vec<u8> = Vec::with_capacity(sweeps_list.len());
    let mut elevation_angles: Vec<f32> = Vec::with_capacity(sweeps_list.len());
    let mut n_radials_list: Vec<usize> = Vec::with_capacity(sweeps_list.len());
    let mut start_indices: Vec<usize> = Vec::with_capacity(sweeps_list.len());

    for sweep_info in sweeps_list.iter() {
        elevation_numbers.push(sweep_info.get_item("elevation_number")?.extract()?);
        elevation_angles.push(sweep_info.get_item("elevation_angle")?.extract()?);
        n_radials_list.push(sweep_info.get_item("n_radials")?.extract()?);
        start_indices.push(sweep_info.get_item("start_index")?.extract()?);
    }

    let sweeps_coords = PyDict::new(py);
    sweeps_coords.set_item("sweep_time", (("sweep_time",), sweep_time_dt))?;
    sweeps_coords.set_item(
        "vcp_time",
        (("sweep_time",), vec![vcp_time; sweeps_list.len()]),
    )?;

    let sweeps_vars = PyDict::new(py);
    // Convert Vec<u8> to Vec<i32> to avoid PyO3 converting to bytes
    let elevation_numbers_i32: Vec<i32> = elevation_numbers.iter().map(|&x| x as i32).collect();
    sweeps_vars.set_item("sweep_number", (("sweep_time",), elevation_numbers_i32))?;
    sweeps_vars.set_item("sweep_fixed_angle", (("sweep_time",), elevation_angles))?;
    sweeps_vars.set_item("n_radials", (("sweep_time",), n_radials_list))?;
    sweeps_vars.set_item("start_index", (("sweep_time",), start_indices))?;

    let sweeps_ds = xr.call_method(
        "Dataset",
        (),
        Some(&[("data_vars", sweeps_vars.as_any()), ("coords", sweeps_coords.as_any())].into_py_dict(py)?),
    )?;

    // Build returns dataset
    let returns_coords = PyDict::new(py);
    returns_coords.set_item("return_time", (("return_time",), return_time_dt))?;

    if let Ok(azimuth_arr) = returns.get_item("azimuth") {
        returns_coords.set_item("azimuth", (("return_time",), azimuth_arr))?;
    }
    if let Ok(elevation_arr) = returns.get_item("elevation") {
        returns_coords.set_item("elevation", (("return_time",), elevation_arr))?;
    }
    if let Ok(sweep_idx_arr) = returns.get_item("sweep_idx") {
        returns_coords.set_item("sweep_idx", (("return_time",), sweep_idx_arr))?;
    }
    returns_coords.set_item("sweep_time", (("return_time",), sweep_time_per_return_dt))?;
    returns_coords.set_item(
        "vcp_time",
        (("return_time",), vec![vcp_time; n_returns]),
    )?;

    // Determine fold size from first available moment
    let moment_names = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"];
    let mut fold_size = None;
    for moment_name in &moment_names {
        if let Ok(moment_arr) = returns.get_item(*moment_name) {
            if let Ok(shape) = moment_arr.getattr("shape") {
                if let Ok(shape_tuple) = shape.extract::<(usize, usize)>() {
                    fold_size = Some(shape_tuple.1);
                    break;
                }
            }
        }
    }

    if let Some(fold) = fold_size {
        let range_data: Vec<usize> = (0..fold).collect();
        returns_coords.set_item("range", (("range",), range_data))?;
    }

    let returns_vars = PyDict::new(py);
    for moment_name in &moment_names {
        if let Ok(moment_arr) = returns.get_item(*moment_name) {
            returns_vars.set_item(*moment_name, (("return_time", "range"), moment_arr))?;
        }
    }

    let returns_ds = xr.call_method(
        "Dataset",
        (),
        Some(&[("data_vars", returns_vars.as_any()), ("coords", returns_coords.as_any())].into_py_dict(py)?),
    )?;

    // Root dataset with minimal attrs
    let root_attrs = PyDict::new(py);
    root_attrs.set_item("Conventions", "CF-1.8")?;
    root_attrs.set_item("instrument_type", "radar")?;
    root_attrs.set_item("volume_coverage_pattern", pattern_number)?;

    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;

    // Build DataTree
    let tree_dict = PyDict::new(py);
    tree_dict.set_item("/", root_ds)?;
    tree_dict.set_item("vcps", vcps_ds)?;
    tree_dict.set_item("sweeps", sweeps_ds)?;
    tree_dict.set_item("returns", returns_ds)?;

    let datatree_class = xr.getattr("DataTree")?;
    let datatree = datatree_class.call_method1("from_dict", (tree_dict,))?;

    Ok(datatree.into())
}
