use rayon::prelude::*;
use std::cmp::Ordering;

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

fn convolve_ones_symm_par(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    window: usize,
) -> Vec<f64> {
    let half = (window / 2) as isize;
    let win = window as isize;
    let total = n_rows * n_cols;

    (0..total)
        .into_par_iter()
        .map(|idx| {
            let r = idx / n_cols;
            let c = idx % n_cols;
            let mut sum = 0.0f64;
            for dr in 0..win {
                let rr = symmetric_index(r as isize + dr - half, n_rows);
                for dc in 0..win {
                    let cc = symmetric_index(c as isize + dc - half, n_cols);
                    sum += data[rr * n_cols + cc];
                }
            }
            sum
        })
        .collect()
}

fn median_filter_symm_par(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    window: usize,
) -> Vec<f64> {
    let half = (window / 2) as isize;
    let win = window as isize;
    let total = n_rows * n_cols;

    (0..total)
        .into_par_iter()
        .map(|idx| {
            let r = idx / n_cols;
            let c = idx % n_cols;
            let mut buf = Vec::with_capacity(window * window);
            for dr in 0..win {
                let rr = reflect_index(r as isize + dr - half, n_rows);
                for dc in 0..win {
                    let cc = reflect_index(c as isize + dc - half, n_cols);
                    buf.push(data[rr * n_cols + cc]);
                }
            }
            // Use select_nth_unstable for O(n) median instead of O(n log n) sort
            let mid = buf.len() / 2;
            let (_, median, _) = buf.select_nth_unstable_by(mid, |a, b| nan_last_cmp(*a, *b));
            *median
        })
        .collect()
}

fn nan_last_cmp(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

fn symmetric_index(mut idx: isize, len: usize) -> usize {
    let len_i = len as isize;
    if len_i == 0 {
        return 0;
    }
    while idx < 0 || idx >= len_i {
        if idx < 0 {
            idx = -idx - 1;
        } else {
            idx = 2 * len_i - idx - 1;
        }
    }
    idx as usize
}

fn reflect_index(mut idx: isize, len: usize) -> usize {
    let len_i = len as isize;
    if len_i == 0 {
        return 0;
    }
    while idx < 0 || idx >= len_i {
        if idx < 0 {
            idx = -idx;
        } else {
            idx = 2 * len_i - idx - 2;
        }
    }
    idx as usize
}
