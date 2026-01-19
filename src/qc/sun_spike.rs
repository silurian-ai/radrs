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
/// * `fill_threshold` - Minimum fraction of non-NaN values to consider (default 0.9)
/// * `corr_threshold` - Minimum autocorrelation for sun spike (default 0.8)
///
/// # Returns
/// int8 mask array: 1=valid, 0=sun spike detected, -1=missing
pub fn sun_spike(
    dbzh: &[f32],
    n_rows: usize,
    n_cols: usize,
    fill_threshold: f32,
    corr_threshold: f32,
) -> Vec<i8> {
    let mut mask = vec![1i8; dbzh.len()];

    // Process each radial (row)
    for row in 0..n_rows {
        let start = row * n_cols;
        let end = start + n_cols;
        let row_data = &dbzh[start..end];

        // Count valid (non-NaN) values
        let valid_count = row_data.iter().filter(|x| !x.is_nan()).count();
        let fill_rate = valid_count as f32 / n_cols as f32;

        // If fill rate is high, check for sun spike pattern
        if fill_rate >= fill_threshold && valid_count > 1 {
            // Calculate autocorrelation at lag 1
            let autocorr = calculate_autocorrelation(row_data);

            if autocorr >= corr_threshold {
                // Mark entire radial as sun spike
                for i in start..end {
                    if dbzh[i].is_nan() {
                        mask[i] = -1;
                    } else {
                        mask[i] = 0; // Sun spike detected
                    }
                }
            } else {
                // Mark NaN values as missing
                for i in start..end {
                    if dbzh[i].is_nan() {
                        mask[i] = -1;
                    }
                }
            }
        } else {
            // Low fill rate - mark NaN as missing, others as valid
            for i in start..end {
                if dbzh[i].is_nan() {
                    mask[i] = -1;
                }
            }
        }
    }

    mask
}

/// Calculate autocorrelation at lag 1
fn calculate_autocorrelation(data: &[f32]) -> f32 {
    // Filter out NaN values
    let valid: Vec<f32> = data.iter().filter(|x| !x.is_nan()).copied().collect();

    if valid.len() < 2 {
        return 0.0;
    }

    // Calculate mean
    let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;

    // Calculate variance
    let variance: f32 = valid.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / valid.len() as f32;

    if variance < 1e-10 {
        return 1.0; // Constant signal has perfect autocorrelation
    }

    // Calculate autocovariance at lag 1
    let mut autocovar: f32 = 0.0;
    for i in 0..valid.len() - 1 {
        autocovar += (valid[i] - mean) * (valid[i + 1] - mean);
    }
    autocovar /= (valid.len() - 1) as f32;

    // Return autocorrelation
    autocovar / variance
}

/// Detect sun spikes (Python wrapper)
#[pyfunction]
#[pyo3(name = "sun_spike", signature = (dbzh, fill_threshold = None, corr_threshold = None))]
pub fn sun_spike_py<'py>(
    py: Python<'py>,
    dbzh: PyReadonlyArray2<'py, f32>,
    fill_threshold: Option<f32>,
    corr_threshold: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let fill_threshold = fill_threshold.unwrap_or(0.9);
    let corr_threshold = corr_threshold.unwrap_or(0.8);

    let shape = dbzh.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    let dbzh_slice = dbzh.as_slice()?;

    let mask_flat = sun_spike(dbzh_slice, n_rows, n_cols, fill_threshold, corr_threshold);

    let mask_arr = mask_flat.into_pyarray(py);
    let mask_2d = mask_arr.reshape([n_rows, n_cols])?;

    Ok(mask_2d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_constant() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let autocorr = calculate_autocorrelation(&data);
        assert!((autocorr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_autocorrelation_with_nan() {
        let data = vec![1.0, 2.0, f32::NAN, 3.0, 4.0];
        let autocorr = calculate_autocorrelation(&data);
        // Should be positive for increasing sequence
        assert!(autocorr > 0.0);
    }

    #[test]
    fn test_sun_spike_detection() {
        // Create a sun spike pattern: constant high values across range
        let mut data = vec![30.0; 100];
        let mask = sun_spike(&data, 1, 100, 0.9, 0.8);

        // Should detect as sun spike (all zeros except NaN)
        assert!(mask.iter().all(|&x| x == 0 || x == -1));
    }
}
