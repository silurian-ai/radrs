//! Conversion between xradar DataTree and unified raystack batch format.

use crate::constants::MOMENT_NAMES;
use crate::metadata_build::{build_root_attrs, set_sweep_mode_scalars, set_sweep_mode_vars};
use crate::raystack::batch::{RaystackBatchData, add_activity_to_dict, compute_batch_activity};
use crate::raystack::parse::DEFAULT_FOLD_SIZE;
use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use std::collections::HashMap;

/// Convert xarray DataTree (xradar) to unified batch raystack format.
#[pyfunction]
#[pyo3(name = "from_xradar_datatree", signature = (datatree, fold_size = None, include_activity = true))]
pub fn from_xradar_datatree_py(
    py: Python<'_>,
    datatree: &Bound<'_, PyAny>,
    fold_size: Option<usize>,
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let np = py.import("numpy")?;

    let children_dict = get_children_dict(py, datatree)?;
    let sweep_meta = collect_sweep_meta(py, datatree, &children_dict)?;

    let max_sweeps = sweep_meta.len().max(1);
    let max_returns: usize = sweep_meta
        .iter()
        .map(|s| {
            let folds_per_radial = if fold_size == 0 {
                1
            } else {
                s.max_gates.div_ceil(fold_size)
            }
            .max(1);
            s.n_radials * folds_per_radial
        })
        .sum::<usize>()
        .max(1);

    let mut batch = RaystackBatchData::new(
        1,
        max_sweeps,
        max_returns,
        fold_size,
        true,
        false,
        true,
        true,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let root_attrs = datatree.getattr("attrs")?;
    let vcp_number: u16 = root_attrs
        .get_item("volume_coverage_pattern")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);
    let instrument_name: Option<String> = root_attrs
        .get_item("instrument_name")
        .ok()
        .and_then(|v| v.extract().ok());

    let mut latitude: Option<f32> = root_attrs
        .get_item("latitude")
        .ok()
        .and_then(|v| v.extract().ok());
    let mut longitude: Option<f32> = root_attrs
        .get_item("longitude")
        .ok()
        .and_then(|v| v.extract().ok());
    let mut altitude: Option<f32> = root_attrs
        .get_item("altitude")
        .ok()
        .and_then(|v| v.extract().ok());

    if (latitude.is_none() || longitude.is_none() || altitude.is_none()) && !sweep_meta.is_empty()
        && let Some(child) = children_dict.get_item(sweep_meta[0].name.as_str())? {
            let dataset = child.getattr("dataset")?;
            latitude = latitude.or_else(|| extract_scalar_coord(&np, &dataset, "latitude"));
            longitude = longitude.or_else(|| extract_scalar_coord(&np, &dataset, "longitude"));
            altitude = altitude.or_else(|| extract_scalar_coord(&np, &dataset, "altitude"));
        }

    let vcp_time = estimate_vcp_time(py, &np, &children_dict, &sweep_meta).unwrap_or(0);

    batch
        .add_vcp_metadata(
            0,
            instrument_name.as_deref(),
            latitude,
            longitude,
            altitude,
            vcp_number,
            vcp_time,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    for meta in &sweep_meta {
        let Some(child) = children_dict.get_item(meta.name.as_str())? else {
            continue;
        };
        let dataset = child.getattr("dataset")?;

        let azimuths = extract_1d_f32(&np, &dataset, "azimuth")?;
        if azimuths.is_empty() {
            continue;
        }
        let n_radials = azimuths.len();

        let elevations = if has_item(&dataset, "elevation")? {
            extract_1d_f32(&np, &dataset, "elevation")?
        } else {
            vec![meta.elevation_angle; n_radials]
        };

        let times = if has_item(&dataset, "time")? {
            extract_1d_time_ms(&np, &dataset, "time")?
        } else {
            vec![vcp_time; n_radials]
        };

        let sweep_time = times.iter().copied().min().unwrap_or(vcp_time);

        let mut moment_buffers: [Option<Vec<f32>>; 7] = std::array::from_fn(|_| None);
        for (idx, moment_name) in MOMENT_NAMES.iter().enumerate() {
            if has_item(&dataset, moment_name)? {
                let values = flatten_f32(&np, &dataset.get_item(*moment_name)?.getattr("values")?)?;
                moment_buffers[idx] = Some(values);
            }
        }

        let moment_refs: [Option<&[f32]>; 7] =
            std::array::from_fn(|idx| moment_buffers[idx].as_deref());

        batch
            .add_sweep_from_arrays(
                vcp_time,
                sweep_time,
                meta.elevation_angle,
                meta.elevation_number,
                meta.range_start_m,
                meta.range_step_m,
                meta.max_gates,
                &azimuths,
                &elevations,
                &times,
                moment_refs,
                n_radials,
                meta.max_gates,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    let activity = if include_activity {
        Some(compute_batch_activity(&batch))
    } else {
        None
    };
    let out = batch.to_python_dict(py)?;
    if let Some(activity) = activity {
        add_activity_to_dict(py, out.bind(py), activity)?;
    }

    Ok(out.into())
}

/// Convert unified raystack dict to xarray DataTree in xradar layout.
#[pyfunction]
#[pyo3(name = "to_xradar_datatree")]
pub fn to_xradar_datatree_py(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    raystack_dict_to_xradar_datatree(py, raystack_dict)
}

/// Convert unified raystack dict to raystack-style DataTree (vcps/sweeps/returns).
#[pyfunction]
#[pyo3(name = "to_raystack_datatree")]
pub fn to_raystack_datatree_py(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    raystack_dict_to_raystack_datatree(py, raystack_dict)
}

#[derive(Clone)]
struct SweepMeta {
    name: String,
    elevation_number: u8,
    elevation_angle: f32,
    n_radials: usize,
    max_gates: usize,
    range_start_m: f32,
    range_step_m: f32,
}

fn get_children_dict<'py>(
    py: Python<'py>,
    datatree: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let children = datatree.getattr("children")?;
    let builtins = py.import("builtins")?;
    let children_dict = builtins.call_method1("dict", (&children,))?;
    let children_dict = children_dict.cast::<PyDict>()?;
    Ok(children_dict.to_owned())
}

fn extract_sweep_number(name: &str) -> Option<usize> {
    name.strip_prefix("sweep_").and_then(|n| n.parse().ok())
}

fn collect_sweep_meta(
    py: Python<'_>,
    datatree: &Bound<'_, PyAny>,
    children_dict: &Bound<'_, PyDict>,
) -> PyResult<Vec<SweepMeta>> {
    let np = py.import("numpy")?;

    let mut sweep_names: Vec<String> = children_dict
        .iter()
        .filter_map(|(k, _)| k.extract::<String>().ok())
        .filter(|name| name.starts_with("sweep_"))
        .collect();
    sweep_names.sort_by(
        |a, b| match (extract_sweep_number(a), extract_sweep_number(b)) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.cmp(b),
        },
    );

    let mut out = Vec::new();

    for name in sweep_names {
        let Some(child) = children_dict.get_item(name.as_str())? else {
            continue;
        };
        let dataset = child.getattr("dataset")?;

        if !has_item(&dataset, "azimuth")? {
            continue;
        }
        let azimuth = extract_1d_f32(&np, &dataset, "azimuth")?;
        let n_radials = azimuth.len();
        if n_radials == 0 {
            continue;
        }

        let attrs = dataset.getattr("attrs")?;

        let parsed_sweep_number = extract_sweep_number(&name).unwrap_or(out.len()) as u32;
        let sweep_number: u32 = attrs
            .get_item("sweep_number")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(parsed_sweep_number);

        let elevation_number: u8 = attrs
            .get_item("elevation_number")
            .ok()
            .and_then(|v| v.extract().ok())
            .or_else(|| {
                attrs
                    .get_item("sweep_number")
                    .ok()
                    .and_then(|v| v.extract().ok())
            })
            .unwrap_or(sweep_number as u8);

        let elevation_angle = attrs
            .get_item("sweep_fixed_angle")
            .ok()
            .and_then(|v| v.extract().ok())
            .or_else(|| {
                if has_item(&dataset, "elevation").ok()? {
                    extract_1d_f32(&np, &dataset, "elevation")
                        .ok()?
                        .first()
                        .copied()
                } else {
                    None
                }
            })
            .unwrap_or(0.0);

        let (max_gates, range_start_m, range_step_m) = if has_item(&dataset, "range")? {
            let range_values = flatten_f64(&np, &dataset.get_item("range")?.getattr("values")?)?;
            if range_values.len() >= 2 {
                (
                    range_values.len(),
                    range_values[0] as f32,
                    (range_values[1] - range_values[0]) as f32,
                )
            } else if range_values.len() == 1 {
                (1, range_values[0] as f32, 0.0)
            } else {
                (0, 0.0, 0.0)
            }
        } else {
            (0, 0.0, 0.0)
        };

        out.push(SweepMeta {
            name,
            elevation_number,
            elevation_angle,
            n_radials,
            max_gates,
            range_start_m,
            range_step_m,
        });
    }

    if out.is_empty() {
        let root_attrs = datatree.getattr("attrs")?;
        let volume_coverage_pattern: Option<u16> = root_attrs
            .get_item("volume_coverage_pattern")
            .ok()
            .and_then(|v| v.extract().ok());
        if volume_coverage_pattern.is_none() {
            return Ok(Vec::new());
        }
    }

    Ok(out)
}

fn has_item(dataset: &Bound<'_, PyAny>, key: &str) -> PyResult<bool> {
    Ok(dataset.get_item(key).is_ok())
}

fn flatten_f32(np: &Bound<'_, PyModule>, values: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let arr = np.call_method1("asarray", (values,))?;
    let flat = np.call_method1("ravel", (&arr,))?;
    flat.extract()
}

fn flatten_f64(np: &Bound<'_, PyModule>, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let arr = np.call_method1("asarray", (values,))?;
    let flat = np.call_method1("ravel", (&arr,))?;
    flat.extract()
}

fn extract_1d_f32(
    np: &Bound<'_, PyModule>,
    dataset: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<f32>> {
    let values = dataset.get_item(name)?.getattr("values")?;
    flatten_f32(np, &values)
}

fn extract_1d_time_ms(
    np: &Bound<'_, PyModule>,
    dataset: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<i64>> {
    let values = dataset.get_item(name)?.getattr("values")?;
    let as_int = values.call_method1("astype", ("int64",))?;
    let flat = np.call_method1("ravel", (&as_int,))?;
    let raw: Vec<i64> = flat.extract()?;
    Ok(raw.into_iter().map(|x| x / 1_000_000).collect())
}

fn extract_scalar_coord(
    np: &Bound<'_, PyModule>,
    dataset: &Bound<'_, PyAny>,
    name: &str,
) -> Option<f32> {
    let coord = dataset.get_item(name).ok()?;
    let values = coord.getattr("values").ok()?;
    let flat = np.call_method1("ravel", (&values,)).ok()?;
    let data: Vec<f32> = flat.extract().ok()?;
    data.first().copied()
}

fn estimate_vcp_time(
    _py: Python<'_>,
    np: &Bound<'_, PyModule>,
    children_dict: &Bound<'_, PyDict>,
    sweep_meta: &[SweepMeta],
) -> Option<i64> {
    let mut min_timestamp: Option<i64> = None;

    for meta in sweep_meta {
        let child = children_dict.get_item(meta.name.as_str()).ok()??;
        let dataset = child.getattr("dataset").ok()?;
        if !has_item(&dataset, "time").ok()? {
            continue;
        }
        let times = extract_1d_time_ms(np, &dataset, "time").ok()?;
        if let Some(sweep_min) = times.into_iter().min() {
            min_timestamp = Some(match min_timestamp {
                Some(current_min) => current_min.min(sweep_min),
                None => sweep_min,
            });
        }
    }

    min_timestamp
}

fn datetime64_ms_to_ns<'py>(
    np: &Bound<'py, PyModule>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = np.call_method1("array", (values,))?;
    let ms = arr.call_method1("astype", ("datetime64[ms]",))?;
    ms.call_method1("astype", ("datetime64[ns]",))
}

fn timedelta64_ms_to_ns<'py>(
    np: &Bound<'py, PyModule>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = np.call_method1("array", (values,))?;
    let ms = arr.call_method1("astype", ("timedelta64[ms]",))?;
    ms.call_method1("astype", ("timedelta64[ns]",))
}

fn vec_first<T: Clone>(v: &[T]) -> Option<T> {
    v.first().cloned()
}

fn raystack_dict_to_xradar_datatree(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    let vcps = raystack_dict.get_item("vcps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'vcps' in raystack dict")
    })?;
    let vcps = vcps.cast::<PyDict>()?;
    let sweeps = raystack_dict.get_item("sweeps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'sweeps' in raystack dict")
    })?;
    let sweeps = sweeps.cast::<PyDict>()?;
    let returns = raystack_dict.get_item("returns")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'returns' in raystack dict")
    })?;
    let returns = returns.cast::<PyDict>()?;

    let vcp_numbers: Vec<u16> = vcps
        .get_item("vcp_number")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing vcps.vcp_number"))?
        .extract()?;
    let pattern_number = vec_first(&vcp_numbers).unwrap_or(0);

    let instrument_names: Vec<String> = if let Some(v) = vcps.get_item("instrument_name")? {
        v.extract().unwrap_or_default()
    } else {
        Vec::new()
    };
    let instrument_name = instrument_names.first().cloned().filter(|s| !s.is_empty());

    let root_attrs = build_root_attrs(py, pattern_number, instrument_name.as_deref())?;

    let tree_dict = PyDict::new(py);
    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;
    tree_dict.set_item("/", root_ds)?;

    let sweep_numbers: Vec<u32> = sweeps
        .get_item("sweep_number")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.sweep_number"))?
        .extract()?;
    let sweep_elevation_numbers: Vec<u8> = sweeps
        .get_item("elevation_number")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.elevation_number"))?
        .extract()?;
    let sweep_elevation_angles: Vec<f32> = sweeps
        .get_item("elevation_angle")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.elevation_angle"))?
        .extract()?;
    let sweep_range_starts: Vec<f32> = sweeps
        .get_item("range_start")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.range_start"))?
        .extract()?;
    let sweep_range_steps: Vec<f32> = sweeps
        .get_item("range_step")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.range_step"))?
        .extract()?;
    let sweep_max_gates: Vec<u32> = sweeps
        .get_item("max_gates")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.max_gates"))?
        .extract()?;

    let return_sweep_numbers: Vec<u32> = returns
        .get_item("sweep_number")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.sweep_number"))?
        .extract()?;
    let return_times: Vec<i64> = returns
        .get_item("return_time")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.return_time"))?
        .extract()?;
    let return_azimuths: Vec<f32> = returns
        .get_item("azimuth")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.azimuth"))?
        .extract()?;
    let return_elevations: Vec<f32> = returns
        .get_item("elevation")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.elevation"))?
        .extract()?;
    let return_base_ranges: Vec<f32> = returns
        .get_item("base_range")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.base_range"))?
        .extract()?;

    let fold_size = returns
        .get_item("range")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.range"))?
        .len()?;

    let mut moment_flat: HashMap<&'static str, Vec<f32>> = HashMap::new();
    for moment in MOMENT_NAMES {
        if let Some(arr) = returns.get_item(moment)? {
            moment_flat.insert(moment, flatten_f32(&np, &arr)?);
        }
    }

    let mut returns_by_sweep: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, sweep_number) in return_sweep_numbers.iter().copied().enumerate() {
        returns_by_sweep.entry(sweep_number).or_default().push(idx);
    }

    for (s_idx, sweep_number) in sweep_numbers.iter().copied().enumerate() {
        let indices = returns_by_sweep
            .get(&sweep_number)
            .cloned()
            .unwrap_or_default();

        #[derive(Clone)]
        struct RadialGroup {
            azimuth: f32,
            elevation: f32,
            time: i64,
            chunk_indices: Vec<usize>,
        }

        let mut radial_groups = Vec::<RadialGroup>::new();
        let mut radial_lookup: HashMap<(u32, i64), usize> = HashMap::new();

        for idx in indices {
            let key = (return_azimuths[idx].to_bits(), return_times[idx]);
            let entry = radial_lookup.entry(key).or_insert_with(|| {
                radial_groups.push(RadialGroup {
                    azimuth: return_azimuths[idx],
                    elevation: return_elevations[idx],
                    time: return_times[idx],
                    chunk_indices: Vec::new(),
                });
                radial_groups.len() - 1
            });
            radial_groups[*entry].chunk_indices.push(idx);
        }

        for radial in &mut radial_groups {
            radial
                .chunk_indices
                .sort_by(|a, b| return_base_ranges[*a].total_cmp(&return_base_ranges[*b]));
        }

        let mut n_range = sweep_max_gates.get(s_idx).copied().unwrap_or(0) as usize;
        let sweep_range_start = sweep_range_starts.get(s_idx).copied().unwrap_or(0.0);
        let sweep_range_step = sweep_range_steps.get(s_idx).copied().unwrap_or(0.0);

        if n_range == 0 {
            for radial in &radial_groups {
                for (chunk_order, &chunk_idx) in radial.chunk_indices.iter().enumerate() {
                    let gate_start = if sweep_range_step > 0.0 {
                        ((return_base_ranges[chunk_idx] - sweep_range_start) / sweep_range_step)
                            .round()
                            .max(0.0) as usize
                    } else {
                        chunk_order * fold_size
                    };
                    n_range = n_range.max(gate_start + fold_size);
                }
            }
        }

        let n_radials = radial_groups.len();

        let azimuth: Vec<f32> = radial_groups.iter().map(|r| r.azimuth).collect();
        let elevation: Vec<f32> = radial_groups.iter().map(|r| r.elevation).collect();
        let times: Vec<i64> = radial_groups.iter().map(|r| r.time).collect();

        let mut moment_2d: HashMap<&'static str, Vec<f32>> = HashMap::new();
        for (moment, flat_values) in &moment_flat {
            let mut matrix = vec![f32::NAN; n_radials * n_range];

            for (radial_idx, radial) in radial_groups.iter().enumerate() {
                for (chunk_order, &chunk_idx) in radial.chunk_indices.iter().enumerate() {
                    let gate_start = if sweep_range_step > 0.0 {
                        ((return_base_ranges[chunk_idx] - sweep_range_start) / sweep_range_step)
                            .round() as isize
                    } else {
                        (chunk_order * fold_size) as isize
                    };
                    if gate_start < 0 {
                        continue;
                    }

                    let src_start = chunk_idx * fold_size;
                    let row_start = radial_idx * n_range;
                    for gate in 0..fold_size {
                        let dst_gate = gate_start as usize + gate;
                        if dst_gate >= n_range {
                            break;
                        }
                        let src_idx = src_start + gate;
                        if src_idx >= flat_values.len() {
                            break;
                        }
                        matrix[row_start + dst_gate] = flat_values[src_idx];
                    }
                }
            }

            moment_2d.insert(moment, matrix);
        }

        let coords = PyDict::new(py);
        coords.set_item("azimuth", (("time",), azimuth.into_pyarray(py)))?;
        coords.set_item("elevation", (("time",), elevation.into_pyarray(py)))?;
        let time_dt = datetime64_ms_to_ns(&np, &times.into_pyarray(py))?;
        coords.set_item("time", (("time",), time_dt))?;

        let range_vals: Vec<f64> = (0..n_range)
            .map(|i| sweep_range_start as f64 + (i as f64 * sweep_range_step as f64))
            .collect();
        coords.set_item("range", (("range",), range_vals.into_pyarray(py)))?;

        let data_vars = PyDict::new(py);
        for moment in MOMENT_NAMES {
            if let Some(values) = moment_2d.get(moment) {
                let arr = Array2::from_shape_vec((n_radials, n_range), values.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                data_vars.set_item(moment, (("time", "range"), arr.into_pyarray(py)))?;
            }
        }

        let dataset = xr.call_method(
            "Dataset",
            (),
            Some(
                &[
                    ("data_vars", data_vars.as_any()),
                    ("coords", coords.as_any()),
                ]
                .into_py_dict(py)?,
            ),
        )?;

        let attrs = PyDict::new(py);
        attrs.set_item("sweep_number", sweep_number)?;
        attrs.set_item(
            "elevation_number",
            sweep_elevation_numbers
                .get(s_idx)
                .copied()
                .unwrap_or(sweep_number as u8),
        )?;
        attrs.set_item(
            "sweep_fixed_angle",
            sweep_elevation_angles.get(s_idx).copied().unwrap_or(0.0),
        )?;
        set_sweep_mode_scalars(&attrs)?;
        dataset.setattr("attrs", attrs)?;

        tree_dict.set_item(format!("/sweep_{}", sweep_number), dataset)?;
    }

    let datatree = xr
        .getattr("DataTree")?
        .call_method1("from_dict", (tree_dict,))?;
    Ok(datatree.into())
}

fn raystack_dict_to_raystack_datatree(
    py: Python<'_>,
    raystack_dict: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    let np = py.import("numpy")?;

    let vcps = raystack_dict.get_item("vcps")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'vcps' in raystack dict")
    })?;
    let vcps = vcps.cast::<PyDict>()?;

    let vcp_numbers: Vec<u16> = vcps
        .get_item("vcp_number")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing vcps.vcp_number"))?
        .extract()?;
    let pattern_number = vec_first(&vcp_numbers).unwrap_or(0);

    let instrument_names: Vec<String> = if let Some(v) = vcps.get_item("instrument_name")? {
        v.extract().unwrap_or_default()
    } else {
        Vec::new()
    };
    let instrument_name = instrument_names.first().cloned().filter(|s| !s.is_empty());

    let root_attrs = build_root_attrs(py, pattern_number, instrument_name.as_deref())?;

    let tree_dict = PyDict::new(py);
    let root_ds = xr.call_method1("Dataset", (PyDict::new(py),))?;
    root_ds.setattr("attrs", root_attrs)?;
    tree_dict.set_item("/", root_ds)?;

    let vcps_coords = PyDict::new(py);
    let vcp_time = vcps
        .get_item("vcp_time")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing vcps.vcp_time"))?;
    let vcp_time_dt = datetime64_ms_to_ns(&np, &vcp_time)?;
    vcps_coords.set_item("vcp_time", (("vcp_time",), vcp_time_dt))?;

    let vcps_vars = PyDict::new(py);
    for (key_obj, value_obj) in vcps.iter() {
        let key: String = key_obj.extract()?;
        if key == "vcp_time" {
            continue;
        }
        if key == "vcp_duration" {
            let value_td = timedelta64_ms_to_ns(&np, &value_obj)?;
            vcps_vars.set_item(key, (("vcp_time",), value_td))?;
            continue;
        }
        vcps_vars.set_item(key, (("vcp_time",), value_obj))?;
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
    tree_dict.set_item("vcps", vcps_ds)?;

    if let Ok(Some(sweeps_any)) = raystack_dict.get_item("sweeps") {
        let sweeps = sweeps_any.cast::<PyDict>()?;
        let sweeps_coords = PyDict::new(py);
        let sweep_time = sweeps
            .get_item("sweep_time")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.sweep_time"))?;
        let sweep_time_dt = datetime64_ms_to_ns(&np, &sweep_time)?;
        sweeps_coords.set_item("sweep_time", (("sweep_time",), sweep_time_dt))?;

        let sweeps_vars = PyDict::new(py);
        for (key_obj, value_obj) in sweeps.iter() {
            let key: String = key_obj.extract()?;
            if key == "sweep_time" {
                continue;
            }
            if key == "sweep_duration" {
                let value_td = timedelta64_ms_to_ns(&np, &value_obj)?;
                sweeps_vars.set_item(key, (("sweep_time",), value_td))?;
            } else if key == "vcp_time" {
                let value_dt = datetime64_ms_to_ns(&np, &value_obj)?;
                sweeps_vars.set_item(key, (("sweep_time",), value_dt))?;
            } else {
                sweeps_vars.set_item(key, (("sweep_time",), value_obj))?;
            }
        }

        if !sweeps_vars.contains("sweep_fixed_angle")?
            && let Some(elevation_angle) = sweeps.get_item("elevation_angle")? {
                sweeps_vars.set_item("sweep_fixed_angle", (("sweep_time",), elevation_angle))?;
            }

        let n_sweeps = sweeps
            .get_item("sweep_time")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing sweeps.sweep_time"))?
            .len()?;
        set_sweep_mode_vars(&sweeps_vars, n_sweeps)?;

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
        tree_dict.set_item("sweeps", sweeps_ds)?;
    }

    if let Ok(Some(returns_any)) = raystack_dict.get_item("returns") {
        let returns = returns_any.cast::<PyDict>()?;
        let returns_coords = PyDict::new(py);
        let return_time = returns.get_item("return_time")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Missing returns.return_time")
        })?;
        let return_time_dt = datetime64_ms_to_ns(&np, &return_time)?;
        returns_coords.set_item("return_time", (("return_time",), return_time_dt.clone()))?;

        if let Some(range) = returns.get_item("range")? {
            returns_coords.set_item("range", (("range",), range))?;
        }

        let n_returns = returns
            .get_item("return_time")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing returns.return_time"))?
            .len()?;
        let fold_size = if let Some(range) = returns.get_item("range")? {
            range.len()?
        } else {
            0
        };

        let returns_vars = PyDict::new(py);
        for (key_obj, value_obj) in returns.iter() {
            let key: String = key_obj.extract()?;
            if key == "return_time" || key == "range" {
                continue;
            }

            if key == "vcp_time" || key == "sweep_time" {
                let value_dt = datetime64_ms_to_ns(&np, &value_obj)?;
                returns_vars.set_item(key, (("return_time",), value_dt))?;
                continue;
            }

            let is_moment = MOMENT_NAMES.iter().any(|m| *m == key) || key.starts_with("qc.");
            if is_moment {
                let arr = np.call_method1("asarray", (&value_obj,))?;
                let reshaped = arr.call_method1("reshape", ((n_returns, fold_size),))?;
                returns_vars.set_item(key, (("return_time", "range"), reshaped))?;
            } else {
                returns_vars.set_item(key, (("return_time",), value_obj))?;
            }
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
        tree_dict.set_item("returns", returns_ds)?;
    }

    if let Ok(Some(activity_any)) = raystack_dict.get_item("activity") {
        let activity = activity_any.cast::<PyDict>()?;
        let activity_coords = PyDict::new(py);
        if let Some(moment) = activity.get_item("moment")? {
            activity_coords.set_item("moment", (("moment",), moment))?;
        }

        if let Ok(Some(returns_any)) = raystack_dict.get_item("returns") {
            let returns = returns_any.cast::<PyDict>()?;
            let return_time = returns.get_item("return_time")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing returns.return_time")
            })?;
            let return_time_dt = datetime64_ms_to_ns(&np, &return_time)?;
            activity_coords.set_item("return_time", (("return_time",), return_time_dt))?;
        }
        if let Ok(Some(sweeps_any)) = raystack_dict.get_item("sweeps") {
            let sweeps = sweeps_any.cast::<PyDict>()?;
            let sweep_time = sweeps.get_item("sweep_time")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing sweeps.sweep_time")
            })?;
            let sweep_time_dt = datetime64_ms_to_ns(&np, &sweep_time)?;
            activity_coords.set_item("sweep_time", (("sweep_time",), sweep_time_dt))?;
        }
        let vcp_time = vcps
            .get_item("vcp_time")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing vcps.vcp_time"))?;
        let vcp_time_dt = datetime64_ms_to_ns(&np, &vcp_time)?;
        activity_coords.set_item("vcp_time", (("vcp_time",), vcp_time_dt))?;

        let activity_vars = PyDict::new(py);
        for (key_obj, value_obj) in activity.iter() {
            let key: String = key_obj.extract()?;
            if key == "moment" {
                continue;
            }
            let dims = match key.as_str() {
                "ray_valid_count" | "ray_valid_fraction" => ("moment", "return_time"),
                "sweep_valid_count" | "sweep_valid_fraction" => ("moment", "sweep_time"),
                "volume_valid_count" | "volume_valid_fraction" => ("moment", "vcp_time"),
                _ => continue,
            };
            activity_vars.set_item(key, (dims, value_obj))?;
        }

        let activity_ds = xr.call_method(
            "Dataset",
            (),
            Some(
                &[
                    ("data_vars", activity_vars.as_any()),
                    ("coords", activity_coords.as_any()),
                ]
                .into_py_dict(py)?,
            ),
        )?;
        tree_dict.set_item("activity", activity_ds)?;
    }

    let datatree = xr
        .getattr("DataTree")?
        .call_method1("from_dict", (tree_dict,))?;
    Ok(datatree.into())
}
