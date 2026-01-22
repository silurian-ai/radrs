use rayon::prelude::*;
use super::filters::{convolve_ones_symm_par, median_filter_symm_par};

pub(crate) fn velocity_texture(
    vel: &[f32],
    n_rows: usize,
    n_cols: usize,
    wind_size: usize,
    nyq: f32,
) -> Vec<f64> {
    let inv = std::f64::consts::PI / nyq as f64;
    let total = n_rows * n_cols;
    let mut x = vec![0.0f64; total];
    let mut y = vec![0.0f64; total];

    // Parallel cos/sin computation (write-in-place to keep deterministic ordering)
    x.par_iter_mut()
        .zip(y.par_iter_mut())
        .zip(vel.par_iter())
        .for_each(|((xv, yv), &v)| {
            let v = v as f64;
            if v.is_nan() {
                *xv = f64::NAN;
                *yv = f64::NAN;
            } else {
                let im = v * inv;
                *xv = im.cos();
                *yv = im.sin();
            }
        });

    let xs = convolve_ones_symm_par(&x, n_rows, n_cols, wind_size);
    let ys = convolve_ones_symm_par(&y, n_rows, n_cols, wind_size);

    let ns = (wind_size * wind_size) as f64;
    let nyq_pi = nyq as f64 / std::f64::consts::PI;

    // Parallel std_dev computation
    let std_dev: Vec<f64> = xs
        .par_iter()
        .zip(ys.par_iter())
        .map(|(&xsum, &ysum)| {
            let xmean = xsum / ns;
            let ymean = ysum / ns;
            let norm = (xmean * xmean + ymean * ymean).sqrt();
            if norm.is_nan() || norm <= 0.0 {
                f64::NAN
            } else {
                (-2.0 * norm.ln()).sqrt() * nyq_pi
            }
        })
        .collect();

    median_filter_symm_par(&std_dev, n_rows, n_cols, wind_size)
}
