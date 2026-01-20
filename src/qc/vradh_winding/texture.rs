use std::cmp::Ordering;

pub(crate) fn velocity_texture(
    vel: &[f32],
    n_rows: usize,
    n_cols: usize,
    wind_size: usize,
    nyq: f32,
) -> Vec<f64> {
    let mut x = vec![0.0f64; n_rows * n_cols];
    let mut y = vec![0.0f64; n_rows * n_cols];
    let inv = std::f64::consts::PI / nyq as f64;

    for i in 0..(n_rows * n_cols) {
        let v = vel[i] as f64;
        if v.is_nan() {
            x[i] = f64::NAN;
            y[i] = f64::NAN;
            continue;
        }
        let im = v * inv;
        x[i] = im.cos();
        y[i] = im.sin();
    }

    let xs = convolve_ones_symm(&x, n_rows, n_cols, wind_size);
    let ys = convolve_ones_symm(&y, n_rows, n_cols, wind_size);

    let ns = (wind_size * wind_size) as f64;
    let mut std_dev = vec![f64::NAN; n_rows * n_cols];
    for i in 0..(n_rows * n_cols) {
        let xmean = xs[i] / ns;
        let ymean = ys[i] / ns;
        let norm = (xmean * xmean + ymean * ymean).sqrt();
        if norm.is_nan() || norm <= 0.0 {
            std_dev[i] = f64::NAN;
        } else {
            std_dev[i] = (-2.0 * norm.ln()).sqrt() * nyq as f64 / std::f64::consts::PI;
        }
    }

    median_filter_symm(&std_dev, n_rows, n_cols, wind_size)
}

fn convolve_ones_symm(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    window: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_rows * n_cols];
    let half = (window / 2) as isize;
    let win = window as isize;

    for r in 0..n_rows {
        for c in 0..n_cols {
            let mut sum = 0.0f64;
            for dr in 0..win {
                let rr = symmetric_index(r as isize + dr - half, n_rows);
                for dc in 0..win {
                    let cc = symmetric_index(c as isize + dc - half, n_cols);
                    let v = data[rr * n_cols + cc];
                    sum += v;
                }
            }
            out[r * n_cols + c] = sum;
        }
    }
    out
}

fn median_filter_symm(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    window: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_rows * n_cols];
    let half = (window / 2) as isize;
    let win = window as isize;
    let mut buf = Vec::with_capacity(window * window);

    for r in 0..n_rows {
        for c in 0..n_cols {
            buf.clear();
            for dr in 0..win {
                let rr = reflect_index(r as isize + dr - half, n_rows);
                for dc in 0..win {
                    let cc = reflect_index(c as isize + dc - half, n_cols);
                    buf.push(data[rr * n_cols + cc]);
                }
            }
            buf.sort_by(|a, b| nan_last_cmp(*a, *b));
            let mid = buf.len() / 2;
            out[r * n_cols + c] = buf[mid];
        }
    }
    out
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
