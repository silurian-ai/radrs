//! VRADH winding number derived from region-based dealiasing (Py-ART port).
//!
//! Note: This is a close port of Py-ART, but rare fold differences can occur due to
//! edge ordering, floating-point rounding, and region merge tie-breaking.

mod edges;
mod regions;
mod trackers;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::ops::velocity_texture;
use edges::edge_sum_and_count;
use regions::{
    find_regions, find_sweep_interval_splits, output_from_labels, region_sizes_and_masked,
};
use trackers::{EdgeTracker, RegionTracker, combine_regions, round_even};

#[derive(Clone, Copy)]
pub struct VradhWindingParams {
    pub nyquist: Option<f32>,
    pub wind_size: usize,
    pub velocity_texture_threshold: f32,
    pub reflectivity_threshold: f32,
    pub interval_splits: usize,
    pub skip_between_rays: usize,
    pub skip_along_ray: usize,
    pub centered: bool,
    pub rays_wrap_around: bool,
    pub fill_value: Option<f32>,
    pub fill_tolerance: f32,
}

impl Default for VradhWindingParams {
    fn default() -> Self {
        Self {
            nyquist: None,
            wind_size: 3,
            velocity_texture_threshold: 4.0,
            reflectivity_threshold: 0.0,
            interval_splits: 3,
            skip_between_rays: 100,
            skip_along_ray: 100,
            centered: true,
            rays_wrap_around: true,
            fill_value: Some(-64.5),
            fill_tolerance: 1.0,
        }
    }
}

/// Compute VRADH winding number for a single sweep (2D arrays).
pub fn vradh_winding_number(
    vradh: &[f32],
    dbzh: Option<&[f32]>,
    n_rows: usize,
    n_cols: usize,
    params: VradhWindingParams,
) -> Vec<f32> {
    let total = n_rows * n_cols;
    if total == 0 {
        return Vec::new();
    }

    let mut vel = Vec::with_capacity(total);
    let mut invalid = Vec::with_capacity(total);
    for &v in vradh.iter().take(total) {
        let mut is_invalid = v.is_nan();
        if let Some(fill) = params.fill_value
            && (v - fill).abs() <= params.fill_tolerance {
                is_invalid = true;
            }
        if is_invalid {
            vel.push(f32::NAN);
        } else {
            vel.push(v);
        }
        invalid.push(is_invalid);
    }

    let nyq = match params.nyquist {
        Some(n) if n > 0.0 => n,
        _ => {
            let mut max_abs = 0.0f32;
            for &v in vel.iter() {
                if v.is_nan() {
                    continue;
                }
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            if max_abs <= 0.0 {
                return vec![f32::NAN; total];
            }
            max_abs
        }
    };

    let wind_size = params.wind_size.max(1);
    let vel_texture = velocity_texture(&vel, n_rows, n_cols, wind_size, nyq);

    // gatefilter: invalid VRADH, high texture, low reflectivity
    let mut gfilter = vec![false; total];
    for i in 0..total {
        if invalid[i] {
            gfilter[i] = true;
            continue;
        }

        let tex = vel_texture[i];
        if !tex.is_nan() && tex > params.velocity_texture_threshold as f64 {
            gfilter[i] = true;
            continue;
        }

        if let Some(dbzh_vals) = dbzh {
            let z = dbzh_vals[i];
            if !z.is_nan() && z < params.reflectivity_threshold {
                gfilter[i] = true;
            }
        }
    }

    // find region labels
    let limits = find_sweep_interval_splits(nyq, params.interval_splits, &vel, &gfilter);
    let (labels, nfeatures) = find_regions(&vel, &gfilter, n_rows, n_cols, &limits);

    // if no regions or single region, winding numbers are zero for valid gates
    if nfeatures < 2 {
        return output_from_labels(&labels, n_rows, n_cols, &gfilter, &vec![0; nfeatures + 1]);
    }

    let (region_sizes, num_masked_gates) =
        region_sizes_and_masked(&labels, nfeatures, n_rows, n_cols);

    let (indices, edge_count, vel_sums) = edge_sum_and_count(
        &labels,
        num_masked_gates,
        &vel,
        n_rows,
        n_cols,
        params.rays_wrap_around,
        params.skip_between_rays,
        params.skip_along_ray,
    );

    if edge_count.is_empty() {
        return output_from_labels(&labels, n_rows, n_cols, &gfilter, &vec![0; nfeatures + 1]);
    }

    let nyq_interval = nyq as f64 * 2.0;
    let mut region_tracker = RegionTracker::new(&region_sizes);
    let mut edge_tracker = EdgeTracker::new(
        &indices,
        &edge_count,
        &vel_sums,
        nyq_interval,
        nfeatures + 1,
    );

    loop {
        if combine_regions(&mut region_tracker, &mut edge_tracker) {
            break;
        }
    }

    if params.centered {
        let gates_dealiased: i32 = region_sizes.iter().sum();
        if gates_dealiased > 0 {
            let mut total_folds: i64 = 0;
            for (i, &size) in region_sizes.iter().enumerate() {
                let unwrap = region_tracker.unwrap_number[i + 1] as i64;
                total_folds += size as i64 * unwrap;
            }
            let sweep_offset = round_even(total_folds as f64 / gates_dealiased as f64);
            if sweep_offset != 0 {
                for u in region_tracker.unwrap_number.iter_mut() {
                    *u -= sweep_offset;
                }
            }
        }
    }

    output_from_labels(
        &labels,
        n_rows,
        n_cols,
        &gfilter,
        &region_tracker.unwrap_number,
    )
}

/// Python wrapper for winding number on a single sweep.
#[pyfunction]
#[pyo3(
    name = "vradh_winding_number",
    signature = (
        vradh,
        dbzh = None,
        nyquist = None,
        wind_size = None,
        velocity_texture_threshold = None,
        reflectivity_threshold = None,
        interval_splits = None,
        skip_between_rays = None,
        skip_along_ray = None,
        centered = None,
        rays_wrap_around = None,
        fill_value = None,
        fill_tolerance = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn vradh_winding_number_py<'py>(
    py: Python<'py>,
    vradh: PyReadonlyArray2<'py, f32>,
    dbzh: Option<PyReadonlyArray2<'py, f32>>,
    nyquist: Option<f32>,
    wind_size: Option<usize>,
    velocity_texture_threshold: Option<f32>,
    reflectivity_threshold: Option<f32>,
    interval_splits: Option<usize>,
    skip_between_rays: Option<usize>,
    skip_along_ray: Option<usize>,
    centered: Option<bool>,
    rays_wrap_around: Option<bool>,
    fill_value: Option<f32>,
    fill_tolerance: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = vradh.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    let vradh_slice = vradh.as_slice()?;
    let dbzh_slice = match dbzh.as_ref() {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let mut params = VradhWindingParams {
        nyquist,
        ..Default::default()
    };
    if let Some(v) = wind_size {
        params.wind_size = v;
    }
    if let Some(v) = velocity_texture_threshold {
        params.velocity_texture_threshold = v;
    }
    if let Some(v) = reflectivity_threshold {
        params.reflectivity_threshold = v;
    }
    if let Some(v) = interval_splits {
        params.interval_splits = v;
    }
    if let Some(v) = skip_between_rays {
        params.skip_between_rays = v;
    }
    if let Some(v) = skip_along_ray {
        params.skip_along_ray = v;
    }
    if let Some(v) = centered {
        params.centered = v;
    }
    if let Some(v) = rays_wrap_around {
        params.rays_wrap_around = v;
    }
    if fill_value.is_some() {
        params.fill_value = fill_value;
    }
    if let Some(v) = fill_tolerance {
        params.fill_tolerance = v;
    }

    let result = vradh_winding_number(vradh_slice, dbzh_slice, n_rows, n_cols, params);
    let arr = result.into_pyarray(py);
    let arr2 = arr.reshape([n_rows, n_cols])?;
    Ok(arr2)
}
