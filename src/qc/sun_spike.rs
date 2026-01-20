//! Sun spike detection and masking

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Detect sun spikes in reflectivity data
///
/// Sun spikes appear as radials with high fill rates of strong reflectivity
/// that are highly correlated across range gates.
///
/// # Arguments
/// * `dbzh` - 2D array of reflectivity values (time, range)
/// * `dbzh_threshold` - Minimum reflectivity to count as filled (default 0.0)
/// * `fill_threshold` - Minimum fraction of filled gates (default 0.9)
/// * `corr_threshold` - Minimum correlation with range index (default 0.8)
///
/// # Returns
/// int8 mask array: 1=valid, 0=sun spike detected, -1=missing
pub fn sun_spike(
    dbzh: &[f32],
    n_rows: usize,
    n_cols: usize,
    dbzh_threshold: f32,
    fill_threshold: f32,
    corr_threshold: f32,
) -> Vec<i8> {
    let mut mask = vec![1i8; dbzh.len()];

    for row in 0..n_rows {
        let start = row * n_cols;
        let end = start + n_cols;
        let row_data = &dbzh[start..end];

        let mut valid_count = 0usize;
        let mut filled_count = 0usize;
        let mut sum_x = 0.0f32;
        let mut sum_r = 0.0f32;

        for (i, &v) in row_data.iter().enumerate() {
            if v.is_nan() {
                mask[start + i] = -1;
                continue;
            }
            valid_count += 1;
            if v >= dbzh_threshold {
                filled_count += 1;
            }
            sum_x += v;
            sum_r += i as f32;
        }

        if valid_count < 3 {
            continue;
        }

        let fill_fraction = filled_count as f32 / valid_count as f32;
        if fill_fraction < fill_threshold {
            continue;
        }

        let mean_x = sum_x / valid_count as f32;
        let mean_r = sum_r / valid_count as f32;

        let mut numerator = 0.0f32;
        let mut denom_x = 0.0f32;
        let mut denom_r = 0.0f32;

        for (i, &v) in row_data.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let dx = v - mean_x;
            let dr = i as f32 - mean_r;
            numerator += dx * dr;
            denom_x += dx * dx;
            denom_r += dr * dr;
        }

        if denom_x <= 0.0 || denom_r <= 0.0 {
            continue;
        }

        let corr = numerator / (denom_x * denom_r).sqrt();
        if corr >= corr_threshold {
            for (i, &v) in row_data.iter().enumerate() {
                if v.is_nan() {
                    continue;
                }
                mask[start + i] = 0;
            }
        }
    }

    mask
}

/// Detect sun spikes (Python wrapper)
#[pyfunction]
#[pyo3(
    name = "sun_spike",
    signature = (dbzh, dbzh_threshold = None, fill_threshold = None, corr_threshold = None)
)]
pub fn sun_spike_py<'py>(
    py: Python<'py>,
    dbzh: PyReadonlyArray2<'py, f32>,
    dbzh_threshold: Option<f32>,
    fill_threshold: Option<f32>,
    corr_threshold: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let dbzh_threshold = dbzh_threshold.unwrap_or(0.0);
    let fill_threshold = fill_threshold.unwrap_or(0.9);
    let corr_threshold = corr_threshold.unwrap_or(0.8);

    let shape = dbzh.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    let dbzh_slice = dbzh.as_slice()?;

    let mask_flat = sun_spike(
        dbzh_slice,
        n_rows,
        n_cols,
        dbzh_threshold,
        fill_threshold,
        corr_threshold,
    );

    let mask_arr = mask_flat.into_pyarray(py);
    let mask_2d = mask_arr.reshape([n_rows, n_cols])?;

    Ok(mask_2d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sun_spike_detection() {
        // Linearly increasing values should be detected
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = sun_spike(&data, 1, 6, 0.0, 0.9, 0.8);

        assert!(mask.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_sun_spike_respects_thresholds() {
        // Below dbzh threshold should not trigger
        let data = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let mask = sun_spike(&data, 1, 6, 0.0, 0.9, 0.8);

        assert!(mask.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_sun_spike_nan_preserved() {
        let data = vec![f32::NAN, 1.0, 2.0, f32::NAN];
        let mask = sun_spike(&data, 1, 4, 0.0, 0.9, 0.8);

        assert_eq!(mask[0], -1);
        assert_eq!(mask[3], -1);
    }
}
