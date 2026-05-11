pub(crate) fn edge_sum_and_count(
    labels: &[i32],
    num_masked_gates: usize,
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    rays_wrap_around: bool,
    max_gap_x: usize,
    max_gap_y: usize,
) -> ((Vec<i32>, Vec<i32>), Vec<i32>, (Vec<f64>, Vec<f64>)) {
    let total_nodes = if rays_wrap_around {
        n_rows * n_cols - num_masked_gates + n_rows * 2
    } else {
        n_rows * n_cols - num_masked_gates
    };
    let mut edges: Vec<(i32, i32, f64, f64)> = Vec::with_capacity(total_nodes * 4);

    let right = n_rows as isize - 1;
    let bottom = n_cols as isize - 1;
    for x in 0..n_rows {
        for y in 0..n_cols {
            let idx = x * n_cols + y;
            let label = labels[idx];
            if label == 0 {
                continue;
            }
            let vel = data[idx] as f64;

            // left
            let mut x_check = x as isize - 1;
            if x_check == -1 && rays_wrap_around {
                x_check = right;
            }
            if x_check != -1 {
                let (neighbor, nvel) = scan_neighbor(
                    labels,
                    data,
                    n_rows,
                    n_cols,
                    x_check,
                    y as isize,
                    -1,
                    0,
                    max_gap_x,
                    rays_wrap_around,
                );
                add_edge(&mut edges, label, neighbor, vel, nvel);
            }

            // right
            let mut x_check = x as isize + 1;
            if x_check == right + 1 && rays_wrap_around {
                x_check = 0;
            }
            if x_check != right + 1 {
                let (neighbor, nvel) = scan_neighbor(
                    labels,
                    data,
                    n_rows,
                    n_cols,
                    x_check,
                    y as isize,
                    1,
                    0,
                    max_gap_x,
                    rays_wrap_around,
                );
                add_edge(&mut edges, label, neighbor, vel, nvel);
            }

            // top
            let y_check = y as isize - 1;
            if y_check != -1 {
                let (neighbor, nvel) = scan_neighbor(
                    labels, data, n_rows, n_cols, x as isize, y_check, 0, -1, max_gap_y, false,
                );
                add_edge(&mut edges, label, neighbor, vel, nvel);
            }

            // bottom
            let y_check = y as isize + 1;
            if y_check != bottom + 1 {
                let (neighbor, nvel) = scan_neighbor(
                    labels, data, n_rows, n_cols, x as isize, y_check, 0, 1, max_gap_y, false,
                );
                add_edge(&mut edges, label, neighbor, vel, nvel);
            }
        }
    }

    if edges.is_empty() {
        return (
            (Vec::new(), Vec::new()),
            Vec::new(),
            (Vec::new(), Vec::new()),
        );
    }

    edges.sort_by(|a, b| {
        let c = a.1.cmp(&b.1);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
        a.0.cmp(&b.0)
    });

    let mut index1 = Vec::new();
    let mut index2 = Vec::new();
    let mut vel1 = Vec::new();
    let mut vel2 = Vec::new();
    let mut count = Vec::new();

    let mut cur = edges[0];
    let mut cur_count = 1i32;
    for edge in edges.iter().skip(1) {
        if edge.0 == cur.0 && edge.1 == cur.1 {
            cur.2 += edge.2;
            cur.3 += edge.3;
            cur_count += 1;
        } else {
            index1.push(cur.0);
            index2.push(cur.1);
            vel1.push(cur.2);
            vel2.push(cur.3);
            count.push(cur_count);
            cur = *edge;
            cur_count = 1;
        }
    }
    index1.push(cur.0);
    index2.push(cur.1);
    vel1.push(cur.2);
    vel2.push(cur.3);
    count.push(cur_count);

    ((index1, index2), count, (vel1, vel2))
}

fn scan_neighbor(
    labels: &[i32],
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    mut x_check: isize,
    mut y_check: isize,
    x_step: isize,
    y_step: isize,
    max_gap: usize,
    wrap_x: bool,
) -> (i32, f64) {
    let right = n_rows as isize - 1;
    let bottom = n_cols as isize - 1;
    let mut neighbor = labels[x_check as usize * n_cols + y_check as usize];
    let mut nvel = data[x_check as usize * n_cols + y_check as usize] as f64;

    if neighbor == 0 {
        for _ in 0..max_gap {
            x_check += x_step;
            y_check += y_step;
            if x_step != 0 {
                if x_check == -1 {
                    if wrap_x {
                        x_check = right;
                    } else {
                        break;
                    }
                }
                if x_check == right + 1 {
                    if wrap_x {
                        x_check = 0;
                    } else {
                        break;
                    }
                }
            }
            if y_step != 0
                && (y_check == -1 || y_check == bottom + 1) {
                    break;
                }
            neighbor = labels[x_check as usize * n_cols + y_check as usize];
            nvel = data[x_check as usize * n_cols + y_check as usize] as f64;
            if neighbor != 0 {
                break;
            }
        }
    }

    (neighbor, nvel)
}

fn add_edge(edges: &mut Vec<(i32, i32, f64, f64)>, label: i32, neighbor: i32, vel: f64, nvel: f64) {
    if neighbor == label || neighbor == 0 {
        return;
    }
    edges.push((label, neighbor, vel, nvel));
}
