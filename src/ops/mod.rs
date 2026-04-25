//! Low-level array ops for alignment and texture computation.

mod filters;
mod texture;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
pub(crate) use texture::velocity_texture;

#[derive(Clone)]
#[pyclass(skip_from_py_object, module = "radrs._radrs.ops")]
pub struct AzimuthPlan {
    indices: Vec<i32>,
    src_len: usize,
    dst_len: usize,
}

#[pymethods]
impl AzimuthPlan {
    #[getter]
    fn src_len(&self) -> usize {
        self.src_len
    }

    #[getter]
    fn dst_len(&self) -> usize {
        self.dst_len
    }

    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.indices.clone().into_pyarray(py)
    }

    #[pyo3(signature = (data, fill=None))]
    fn apply_1d<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f32>,
        fill: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let len = data.shape()[0];
        if len != self.src_len {
            return Err(PyValueError::new_err(format!(
                "Expected data length {}, got {}",
                self.src_len, len
            )));
        }
        let fill = fill.unwrap_or(f32::NAN);
        let src = data.as_slice()?;
        let mut out = vec![fill; self.dst_len];
        for (dst_idx, &src_idx_i) in self.indices.iter().enumerate() {
            if src_idx_i < 0 {
                continue;
            }
            let src_idx = src_idx_i as usize;
            out[dst_idx] = src[src_idx];
        }
        Ok(out.into_pyarray(py))
    }

    #[pyo3(signature = (data, fill=None))]
    fn apply_2d<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        fill: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = data.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("data must be 2D"));
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        if n_rows != self.src_len {
            return Err(PyValueError::new_err(format!(
                "Expected data shape ({}, N), got ({}, {})",
                self.src_len, n_rows, n_cols
            )));
        }

        let fill = fill.unwrap_or(f32::NAN);
        let src = data.as_slice()?;
        let mut out = vec![fill; self.dst_len * n_cols];
        for (dst_idx, &src_idx_i) in self.indices.iter().enumerate() {
            if src_idx_i < 0 {
                continue;
            }
            let src_idx = src_idx_i as usize;
            let src_offset = src_idx * n_cols;
            let dst_offset = dst_idx * n_cols;
            out[dst_offset..dst_offset + n_cols]
                .copy_from_slice(&src[src_offset..src_offset + n_cols]);
        }

        let arr = out.into_pyarray(py);
        Ok(arr.reshape([self.dst_len, n_cols])?)
    }
}

#[derive(Clone)]
#[pyclass(skip_from_py_object, module = "radrs._radrs.ops")]
pub struct RangePlan {
    indices: Vec<i32>,
    src_len: usize,
    dst_len: usize,
}

#[pymethods]
impl RangePlan {
    #[getter]
    fn src_len(&self) -> usize {
        self.src_len
    }

    #[getter]
    fn dst_len(&self) -> usize {
        self.dst_len
    }

    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.indices.clone().into_pyarray(py)
    }

    #[pyo3(signature = (data, fill=None))]
    fn apply_1d<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f32>,
        fill: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let len = data.shape()[0];
        if len != self.src_len {
            return Err(PyValueError::new_err(format!(
                "Expected data length {}, got {}",
                self.src_len, len
            )));
        }
        let fill = fill.unwrap_or(f32::NAN);
        let src = data.as_slice()?;
        let mut out = vec![fill; self.dst_len];
        for (dst_idx, &src_idx_i) in self.indices.iter().enumerate() {
            if src_idx_i < 0 {
                continue;
            }
            let src_idx = src_idx_i as usize;
            out[dst_idx] = src[src_idx];
        }
        Ok(out.into_pyarray(py))
    }

    #[pyo3(signature = (data, fill=None))]
    fn apply_2d<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        fill: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = data.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("data must be 2D"));
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        if n_cols != self.src_len {
            return Err(PyValueError::new_err(format!(
                "Expected data shape (N, {}), got ({}, {})",
                self.src_len, n_rows, n_cols
            )));
        }

        let fill = fill.unwrap_or(f32::NAN);
        let src = data.as_slice()?;
        let mut out = vec![fill; n_rows * self.dst_len];
        for row in 0..n_rows {
            let src_row_offset = row * n_cols;
            let dst_row_offset = row * self.dst_len;
            for (dst_idx, &src_idx_i) in self.indices.iter().enumerate() {
                if src_idx_i < 0 {
                    continue;
                }
                let src_idx = src_idx_i as usize;
                out[dst_row_offset + dst_idx] = src[src_row_offset + src_idx];
            }
        }

        let arr = out.into_pyarray(py);
        Ok(arr.reshape([n_rows, self.dst_len])?)
    }
}

fn default_azimuth_tolerance(dst_len: usize) -> f32 {
    if dst_len < 2 {
        180.0
    } else {
        360.0 / dst_len as f32 * 0.55
    }
}

fn default_range_tolerance(src_step: f32, dst_step: f32) -> f32 {
    let src = src_step.abs();
    let dst = dst_step.abs();
    let min_step = if src < dst { src } else { dst };
    min_step * 0.55
}

fn azimuth_distance(a: f32, b: f32, wrap: bool) -> f32 {
    let mut diff = (a - b).abs();
    if wrap {
        diff = diff % 360.0;
        let wrapped = 360.0 - diff;
        if wrapped < diff { wrapped } else { diff }
    } else {
        diff
    }
}

fn align_azimuth_indices(
    src_az: &[f32],
    dst_az: &[f32],
    tolerance_deg: f32,
    wrap: bool,
) -> Vec<i32> {
    let mut indices = vec![-1; dst_az.len()];
    if src_az.is_empty() || dst_az.is_empty() {
        return indices;
    }

    for (dst_idx, &dst_val) in dst_az.iter().enumerate() {
        if dst_val.is_nan() {
            continue;
        }
        let mut best_idx: i32 = -1;
        let mut best_diff = f32::INFINITY;
        for (src_idx, &src_val) in src_az.iter().enumerate() {
            if src_val.is_nan() {
                continue;
            }
            let diff = azimuth_distance(src_val, dst_val, wrap);
            if diff < best_diff {
                best_diff = diff;
                best_idx = src_idx as i32;
            }
        }
        if best_idx >= 0 && best_diff <= tolerance_deg {
            indices[dst_idx] = best_idx;
        }
    }

    indices
}

fn align_range_indices(
    src_start_m: f32,
    src_step_m: f32,
    src_len: usize,
    dst_start_m: f32,
    dst_step_m: f32,
    dst_len: usize,
    tolerance_m: f32,
) -> Vec<i32> {
    let mut indices = vec![-1; dst_len];
    if src_len == 0 || dst_len == 0 {
        return indices;
    }

    for dst_idx in 0..dst_len {
        let dst_range = dst_start_m + dst_idx as f32 * dst_step_m;
        let src_pos = (dst_range - src_start_m) / src_step_m;
        let src_idx_round = src_pos.round();
        if !src_idx_round.is_finite() {
            continue;
        }
        let src_idx_i = src_idx_round as i32;
        if src_idx_i < 0 || src_idx_i >= src_len as i32 {
            continue;
        }
        let src_range = src_start_m + src_idx_round * src_step_m;
        if (src_range - dst_range).abs() <= tolerance_m {
            indices[dst_idx] = src_idx_i;
        }
    }

    indices
}

/// Build an azimuth alignment plan (maps dst azimuths to src azimuths).
#[pyfunction]
#[pyo3(name = "align_azimuth", signature = (src_azimuth, dst_azimuth, tolerance_deg=None, wrap=true))]
pub fn align_azimuth_py(
    src_azimuth: PyReadonlyArray1<'_, f32>,
    dst_azimuth: PyReadonlyArray1<'_, f32>,
    tolerance_deg: Option<f32>,
    wrap: bool,
) -> PyResult<AzimuthPlan> {
    let src = src_azimuth.as_slice()?;
    let dst = dst_azimuth.as_slice()?;
    let tol = tolerance_deg.unwrap_or_else(|| default_azimuth_tolerance(dst.len()));
    if tol <= 0.0 {
        return Err(PyValueError::new_err("tolerance_deg must be > 0"));
    }
    let indices = align_azimuth_indices(src, dst, tol, wrap);
    Ok(AzimuthPlan {
        indices,
        src_len: src.len(),
        dst_len: dst.len(),
    })
}

/// Build a range alignment plan (maps dst gates to src gates).
#[pyfunction]
#[pyo3(name = "align_range", signature = (
    src_start_m,
    src_step_m,
    src_len,
    dst_start_m,
    dst_step_m,
    dst_len,
    tolerance_m=None
))]
pub fn align_range_py(
    src_start_m: f32,
    src_step_m: f32,
    src_len: usize,
    dst_start_m: f32,
    dst_step_m: f32,
    dst_len: usize,
    tolerance_m: Option<f32>,
) -> PyResult<RangePlan> {
    if src_step_m == 0.0 || dst_step_m == 0.0 {
        return Err(PyValueError::new_err(
            "src_step_m and dst_step_m must be non-zero",
        ));
    }
    let tol = tolerance_m.unwrap_or_else(|| default_range_tolerance(src_step_m, dst_step_m));
    if tol <= 0.0 {
        return Err(PyValueError::new_err("tolerance_m must be > 0"));
    }
    let indices = align_range_indices(
        src_start_m,
        src_step_m,
        src_len,
        dst_start_m,
        dst_step_m,
        dst_len,
        tol,
    );
    Ok(RangePlan {
        indices,
        src_len,
        dst_len,
    })
}

/// Compute velocity texture for a single sweep (2D array).
#[pyfunction]
#[pyo3(name = "velocity_texture", signature = (vradh, nyquist=None, wind_size=None))]
pub fn velocity_texture_py<'py>(
    py: Python<'py>,
    vradh: PyReadonlyArray2<'py, f32>,
    nyquist: Option<f32>,
    wind_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = vradh.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    let vel = vradh.as_slice()?;

    let mut nyq = nyquist.unwrap_or(0.0);
    if nyq <= 0.0 {
        for &v in vel.iter() {
            if v.is_nan() {
                continue;
            }
            let a = v.abs();
            if a > nyq {
                nyq = a;
            }
        }
    }

    if nyq <= 0.0 {
        let out = vec![f32::NAN; n_rows * n_cols];
        let arr = out.into_pyarray(py);
        return Ok(arr.reshape([n_rows, n_cols])?);
    }

    let window = wind_size.unwrap_or(3).max(1);
    let tex = velocity_texture(vel, n_rows, n_cols, window, nyq);
    let out: Vec<f32> = tex.into_iter().map(|v| v as f32).collect();
    let arr = out.into_pyarray(py);
    Ok(arr.reshape([n_rows, n_cols])?)
}

/// Register ops submodule.
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let ops_module = PyModule::new(parent_module.py(), "ops")?;
    ops_module.add_class::<AzimuthPlan>()?;
    ops_module.add_class::<RangePlan>()?;
    ops_module.add_function(wrap_pyfunction!(align_azimuth_py, &ops_module)?)?;
    ops_module.add_function(wrap_pyfunction!(align_range_py, &ops_module)?)?;
    ops_module.add_function(wrap_pyfunction!(velocity_texture_py, &ops_module)?)?;
    parent_module.add_submodule(&ops_module)?;
    Ok(())
}
