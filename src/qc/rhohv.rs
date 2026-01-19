//! RHOHV (correlation coefficient) threshold-based QC

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Apply RHOHV threshold mask
///
/// # Arguments
/// * `rhohv` - 2D array of RHOHV values (time, range)
/// * `threshold` - Minimum valid RHOHV value (default 0.8)
///
/// # Returns
/// int8 mask array: 1=valid, 0=invalid, -1=missing
pub fn rhohv_threshold(rhohv: &[f32], threshold: f32) -> Vec<i8> {
    rhohv
        .iter()
        .map(|&v| {
            if v.is_nan() {
                -1i8 // Missing
            } else if v >= threshold {
                1i8 // Valid
            } else {
                0i8 // Invalid (below threshold)
            }
        })
        .collect()
}

/// Apply RHOHV threshold mask (Python wrapper)
#[pyfunction]
#[pyo3(name = "rhohv_threshold", signature = (rhohv, threshold = None))]
pub fn rhohv_threshold_py<'py>(
    py: Python<'py>,
    rhohv: PyReadonlyArray2<'py, f32>,
    threshold: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let threshold = threshold.unwrap_or(0.8);
    let shape = rhohv.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    // Get slice of the array
    let rhohv_slice = rhohv.as_slice()?;

    // Apply threshold
    let mask_flat = rhohv_threshold(rhohv_slice, threshold);

    // Convert to 2D numpy array
    let mask_arr = mask_flat.into_pyarray(py);
    let mask_2d = mask_arr
        .reshape([n_rows, n_cols])?;

    Ok(mask_2d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhohv_threshold() {
        let data = vec![0.5, 0.8, 0.9, f32::NAN, 0.7];
        let mask = rhohv_threshold(&data, 0.8);

        assert_eq!(mask[0], 0); // 0.5 < 0.8 -> invalid
        assert_eq!(mask[1], 1); // 0.8 >= 0.8 -> valid
        assert_eq!(mask[2], 1); // 0.9 >= 0.8 -> valid
        assert_eq!(mask[3], -1); // NaN -> missing
        assert_eq!(mask[4], 0); // 0.7 < 0.8 -> invalid
    }
}
