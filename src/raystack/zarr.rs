//! Zarr I/O for raystack format

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Write raystack to Zarr store
#[pyfunction]
#[pyo3(name = "to_zarr", signature = (raystack, path, mode = None))]
pub fn to_zarr_py(
    py: Python<'_>,
    raystack: &Bound<'_, PyDict>,
    path: &str,
    mode: Option<&str>,
) -> PyResult<()> {
    let zarr = py.import("zarr")?;
    let np = py.import("numpy")?;

    let mode = mode.unwrap_or("w");

    // Open or create Zarr store
    let store = zarr.call_method1("open_group", (path,))?;
    store.call_method1("require_group", ("vcps",))?;
    store.call_method1("require_group", ("sweeps",))?;
    store.call_method1("require_group", ("returns",))?;

    // Write VCPs
    let vcps = raystack.get_item("vcps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'vcps'")
    })?;
    let vcps_group = store.call_method1("require_group", ("vcps",))?;
    let pattern_number: u16 = vcps.get_item("pattern_number")?.extract()?;
    vcps_group.setattr("attrs", {
        let d = PyDict::new(py);
        d.set_item("pattern_number", pattern_number)?;
        d
    })?;

    // Write sweeps as metadata
    let sweeps = raystack.get_item("sweeps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'sweeps'")
    })?;
    let sweeps_list: Vec<&Bound<'_, PyAny>> = sweeps.extract()?;
    let sweeps_group = store.call_method1("require_group", ("sweeps",))?;

    for (i, sweep) in sweeps_list.iter().enumerate() {
        let sweep_attrs = PyDict::new(py);
        sweep_attrs.set_item("elevation_number", sweep.get_item("elevation_number")?)?;
        sweep_attrs.set_item("elevation_angle", sweep.get_item("elevation_angle")?)?;
        sweep_attrs.set_item("n_radials", sweep.get_item("n_radials")?)?;
        sweep_attrs.set_item("start_index", sweep.get_item("start_index")?)?;

        let sweep_group = sweeps_group.call_method1("require_group", (format!("sweep_{}", i),))?;
        sweep_group.setattr("attrs", sweep_attrs)?;
    }

    // Write returns arrays
    let returns = raystack.get_item("returns")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'returns'")
    })?;
    let returns_group = store.call_method1("require_group", ("returns",))?;

    // Write coordinate arrays
    for key in ["azimuth", "elevation", "time", "sweep_idx"] {
        if let Ok(arr) = returns.get_item(key) {
            let np_arr = np.call_method1("asarray", (&arr,))?;

            if mode == "a" {
                // Append mode - resize and append
                if let Ok(existing) = returns_group.get_item(key) {
                    let existing_len: usize = existing.getattr("shape")?.get_item(0)?.extract()?;
                    let new_len: usize = np_arr.getattr("shape")?.get_item(0)?.extract()?;
                    let total_len = existing_len + new_len;

                    existing.call_method1("resize", (total_len,))?;
                    existing.set_item(
                        py.eval(&format!("slice({}, {})", existing_len, total_len), None, None)?,
                        &np_arr,
                    )?;
                } else {
                    returns_group.call_method(
                        "create_dataset",
                        (key, np_arr),
                        Some(&[("chunks", (10000,))].into_py_dict(py)?),
                    )?;
                }
            } else {
                returns_group.call_method(
                    "create_dataset",
                    (key, np_arr),
                    Some(&[("chunks", (10000,))].into_py_dict(py)?),
                )?;
            }
        }
    }

    // Write moment arrays
    let moment_names = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"];
    for moment_name in &moment_names {
        if let Ok(arr) = returns.get_item(*moment_name) {
            let np_arr = np.call_method1("asarray", (&arr,))?;
            let shape: (usize, usize) = np_arr.getattr("shape")?.extract()?;

            if mode == "a" {
                if let Ok(existing) = returns_group.get_item(*moment_name) {
                    let existing_shape: (usize, usize) = existing.getattr("shape")?.extract()?;
                    let new_shape = (existing_shape.0 + shape.0, shape.1);

                    existing.call_method1("resize", (new_shape,))?;
                    existing.set_item(
                        py.eval(
                            &format!("(slice({}, {}), slice(None))", existing_shape.0, new_shape.0),
                            None,
                            None,
                        )?,
                        &np_arr,
                    )?;
                } else {
                    returns_group.call_method(
                        "create_dataset",
                        (*moment_name, np_arr),
                        Some(&[("chunks", (1000, shape.1))].into_py_dict(py)?),
                    )?;
                }
            } else {
                returns_group.call_method(
                    "create_dataset",
                    (*moment_name, np_arr),
                    Some(&[("chunks", (1000, shape.1))].into_py_dict(py)?),
                )?;
            }
        }
    }

    Ok(())
}

/// Open a Zarr store for appending raystack data
#[pyfunction]
#[pyo3(name = "open_zarr", signature = (path, mode = None))]
pub fn open_zarr(py: Python<'_>, path: &str, mode: Option<&str>) -> PyResult<PyObject> {
    let zarr = py.import("zarr")?;
    let mode = mode.unwrap_or("r");

    let store = zarr.call_method(
        "open_group",
        (path,),
        Some(&[("mode", mode)].into_py_dict(py)?),
    )?;

    Ok(store.into())
}
