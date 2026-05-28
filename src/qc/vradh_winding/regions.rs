pub(crate) fn find_sweep_interval_splits(
    nyquist: f32,
    interval_splits: usize,
    vel: &[f32],
    gfilter: &[bool],
) -> Vec<f32> {
    let interval = (2.0 * nyquist) / interval_splits as f32;
    let mut add_start = 0i32;
    let mut add_end = 0i32;

    let mut max_vel = f32::NEG_INFINITY;
    let mut min_vel = f32::INFINITY;
    for (i, &v) in vel.iter().enumerate() {
        if gfilter[i] || v.is_nan() {
            continue;
        }
        if v > max_vel {
            max_vel = v;
        }
        if v < min_vel {
            min_vel = v;
        }
    }

    if max_vel.is_finite() && min_vel.is_finite() {
        if max_vel > nyquist {
            add_start = ((max_vel - nyquist) / interval).ceil() as i32;
        }
        if min_vel < -nyquist {
            add_end = (-(min_vel + nyquist) / interval).ceil() as i32;
        }
    }

    let start = -nyquist - add_start as f32 * interval;
    let end = nyquist + add_end as f32 * interval;
    let num = interval_splits as i32 + 1 + add_start + add_end;
    let num = num.max(2) as usize;

    let mut limits = Vec::with_capacity(num);
    if num == 1 {
        limits.push(start);
        return limits;
    }
    let step = (end - start) / (num as f32 - 1.0);
    for i in 0..num {
        limits.push(start + i as f32 * step);
    }
    limits
}

pub(crate) fn find_regions(
    vel: &[f32],
    gfilter: &[bool],
    n_rows: usize,
    n_cols: usize,
    limits: &[f32],
) -> (Vec<i32>, usize) {
    let mut labels = vec![0i32; n_rows * n_cols];
    let mut nfeatures = 0usize;

    for w in limits.windows(2) {
        let lmin = w[0];
        let lmax = w[1];
        let mut local = vec![false; n_rows * n_cols];
        for idx in 0..(n_rows * n_cols) {
            if gfilter[idx] {
                continue;
            }
            let v = vel[idx];
            if v.is_nan() {
                continue;
            }
            if v >= lmin && v < lmax {
                local[idx] = true;
            }
        }

        for r in 0..n_rows {
            for c in 0..n_cols {
                let idx = r * n_cols + c;
                if !local[idx] || labels[idx] != 0 {
                    continue;
                }
                nfeatures += 1;
                let label = nfeatures as i32;
                flood_fill(idx, label, n_rows, n_cols, &local, &mut labels);
            }
        }
    }

    (labels, nfeatures)
}

fn flood_fill(
    start_idx: usize,
    label: i32,
    n_rows: usize,
    n_cols: usize,
    local: &[bool],
    labels: &mut [i32],
) {
    let mut stack = Vec::new();
    stack.push(start_idx);
    labels[start_idx] = label;

    while let Some(idx) = stack.pop() {
        let r = idx / n_cols;
        let c = idx % n_cols;
        let neighbors = [
            (r.wrapping_sub(1), c, r > 0),
            (r + 1, c, r + 1 < n_rows),
            (r, c.wrapping_sub(1), c > 0),
            (r, c + 1, c + 1 < n_cols),
        ];

        for (nr, nc, ok) in neighbors.iter() {
            if !ok {
                continue;
            }
            let nidx = nr * n_cols + nc;
            if local[nidx] && labels[nidx] == 0 {
                labels[nidx] = label;
                stack.push(nidx);
            }
        }
    }
}

pub(crate) fn region_sizes_and_masked(
    labels: &[i32],
    nfeatures: usize,
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i32>, usize) {
    let mut sizes = vec![0i32; nfeatures];
    let mut masked = 0usize;
    for &l in labels.iter().take(n_rows * n_cols) {
        if l == 0 {
            masked += 1;
        } else {
            let i = (l - 1) as usize;
            sizes[i] += 1;
        }
    }
    (sizes, masked)
}

pub(crate) fn output_from_labels(
    labels: &[i32],
    n_rows: usize,
    n_cols: usize,
    gfilter: &[bool],
    unwrap_number: &[i32],
) -> Vec<f32> {
    let mut out = vec![f32::NAN; n_rows * n_cols];
    for idx in 0..(n_rows * n_cols) {
        if gfilter[idx] {
            continue;
        }
        let label = labels[idx] as usize;
        if label < unwrap_number.len() {
            out[idx] = unwrap_number[label] as f32;
        } else {
            out[idx] = 0.0;
        }
    }
    out
}
