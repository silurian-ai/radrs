pub(crate) fn round_even(value: f64) -> i32 {
    if !value.is_finite() {
        return 0;
    }
    let floor = value.floor();
    let frac = value - floor;
    if (frac - 0.5).abs() < 1e-12 {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 {
            floor_i as i32
        } else {
            (floor_i + 1) as i32
        }
    } else if (frac + 0.5).abs() < 1e-12 {
        let ceil_i = value.ceil() as i64;
        if ceil_i % 2 == 0 {
            ceil_i as i32
        } else {
            (ceil_i - 1) as i32
        }
    } else {
        value.round() as i32
    }
}

pub(crate) fn combine_regions(
    region_tracker: &mut RegionTracker,
    edge_tracker: &mut EdgeTracker,
) -> bool {
    let (done, extra) = edge_tracker.pop_edge();
    if done {
        return true;
    }
    let (node1, node2, diff, edge_number) = extra;
    let mut rdiff = round_even(diff as f64);

    let node1_size = region_tracker.get_node_size(node1);
    let node2_size = region_tracker.get_node_size(node2);

    let (base_node, merge_node) = if node1_size > node2_size {
        (node1, node2)
    } else {
        rdiff = -rdiff;
        (node2, node1)
    };

    if rdiff != 0 {
        region_tracker.unwrap_node(merge_node, rdiff);
        edge_tracker.unwrap_node(merge_node, rdiff);
    }

    region_tracker.merge_nodes(base_node, merge_node);
    edge_tracker.merge_nodes(base_node, merge_node, edge_number);

    false
}

pub(crate) struct RegionTracker {
    node_size: Vec<i32>,
    regions_in_node: Vec<Vec<usize>>,
    pub(crate) unwrap_number: Vec<i32>,
}

impl RegionTracker {
    pub(crate) fn new(region_sizes: &[i32]) -> Self {
        let nregions = region_sizes.len() + 1;
        let mut node_size = vec![0i32; nregions];
        for (i, &size) in region_sizes.iter().enumerate() {
            node_size[i + 1] = size;
        }

        let mut regions_in_node = Vec::with_capacity(nregions);
        for i in 0..nregions {
            regions_in_node.push(vec![i]);
        }

        let unwrap_number = vec![0i32; nregions];
        Self {
            node_size,
            regions_in_node,
            unwrap_number,
        }
    }

    pub(crate) fn merge_nodes(&mut self, node_a: usize, node_b: usize) {
        let regions_to_merge = self.regions_in_node[node_b].clone();
        self.regions_in_node[node_a].extend(regions_to_merge);
        self.regions_in_node[node_b].clear();

        self.node_size[node_a] += self.node_size[node_b];
        self.node_size[node_b] = 0;
    }

    pub(crate) fn unwrap_node(&mut self, node: usize, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        let regions_to_unwrap = self.regions_in_node[node].clone();
        for region in regions_to_unwrap {
            self.unwrap_number[region] += nwrap;
        }
    }

    pub(crate) fn get_node_size(&self, node: usize) -> i32 {
        self.node_size[node]
    }
}

pub(crate) struct EdgeTracker {
    node_alpha: Vec<usize>,
    node_beta: Vec<usize>,
    sum_diff: Vec<f32>,
    weight: Vec<i32>,
    edges_in_node: Vec<Vec<usize>>,
    common_finder: Vec<bool>,
    common_index: Vec<usize>,
    last_base_node: isize,
}

impl EdgeTracker {
    pub(crate) fn new(
        indices: &(Vec<i32>, Vec<i32>),
        edge_count: &[i32],
        velocities: &(Vec<f64>, Vec<f64>),
        nyquist_interval: f64,
        nnodes: usize,
    ) -> Self {
        let nedges = indices.0.len() / 2;
        let mut node_alpha = vec![0usize; nedges];
        let mut node_beta = vec![0usize; nedges];
        let mut sum_diff = vec![0.0f32; nedges];
        let mut weight = vec![0i32; nedges];
        let mut edges_in_node: Vec<Vec<usize>> = (0..nnodes).map(|_| Vec::new()).collect();

        let mut edge_idx = 0usize;
        for i in 0..indices.0.len() {
            let a = indices.0[i] as isize;
            let b = indices.1[i] as isize;
            if a < b {
                continue;
            }
            if edge_idx >= nedges {
                break;
            }
            node_alpha[edge_idx] = a as usize;
            node_beta[edge_idx] = b as usize;
            let diff = (velocities.0[i] - velocities.1[i]) / nyquist_interval;
            sum_diff[edge_idx] = diff as f32;
            weight[edge_idx] = edge_count[i];
            edges_in_node[a as usize].push(edge_idx);
            edges_in_node[b as usize].push(edge_idx);
            edge_idx += 1;
        }

        Self {
            node_alpha,
            node_beta,
            sum_diff,
            weight,
            edges_in_node,
            common_finder: vec![false; nnodes],
            common_index: vec![0usize; nnodes],
            last_base_node: -1,
        }
    }

    pub(crate) fn merge_nodes(&mut self, base_node: usize, merge_node: usize, edge_between: usize) {
        self.weight[edge_between] = -999;
        remove_edge(&mut self.edges_in_node[merge_node], edge_between);
        remove_edge(&mut self.edges_in_node[base_node], edge_between);
        self.common_finder[merge_node] = false;

        let edges_in_merge = self.edges_in_node[merge_node].clone();

        if self.last_base_node != base_node as isize {
            self.common_finder.fill(false);
            let edges_in_base = self.edges_in_node[base_node].clone();
            for edge_num in edges_in_base {
                if self.node_beta[edge_num] == base_node {
                    self.reverse_edge_direction(edge_num);
                }
                let neighbor = self.node_beta[edge_num];
                self.common_finder[neighbor] = true;
                self.common_index[neighbor] = edge_num;
            }
        }

        for edge_num in edges_in_merge {
            if self.node_beta[edge_num] == merge_node {
                self.reverse_edge_direction(edge_num);
            }
            self.node_alpha[edge_num] = base_node;
            let neighbor = self.node_beta[edge_num];
            if self.common_finder[neighbor] {
                let base_edge_num = self.common_index[neighbor];
                self.combine_edges(base_edge_num, edge_num, merge_node, neighbor);
            } else {
                self.common_finder[neighbor] = true;
                self.common_index[neighbor] = edge_num;
            }
        }

        let edges = self.edges_in_node[merge_node].clone();
        self.edges_in_node[base_node].extend(edges);
        self.edges_in_node[merge_node].clear();
        self.last_base_node = base_node as isize;
    }

    fn combine_edges(
        &mut self,
        base_edge: usize,
        merge_edge: usize,
        merge_node: usize,
        neighbor_node: usize,
    ) {
        self.weight[base_edge] += self.weight[merge_edge];
        self.weight[merge_edge] = -999;
        self.sum_diff[base_edge] += self.sum_diff[merge_edge];
        remove_edge(&mut self.edges_in_node[merge_node], merge_edge);
        remove_edge(&mut self.edges_in_node[neighbor_node], merge_edge);
    }

    fn reverse_edge_direction(&mut self, edge: usize) {
        let old_alpha = self.node_alpha[edge];
        let old_beta = self.node_beta[edge];
        self.node_alpha[edge] = old_beta;
        self.node_beta[edge] = old_alpha;
        self.sum_diff[edge] = -self.sum_diff[edge];
    }

    pub(crate) fn unwrap_node(&mut self, node: usize, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        let edges = self.edges_in_node[node].clone();
        for edge in edges {
            let weight = self.weight[edge];
            if weight < 0 {
                continue;
            }
            if node == self.node_alpha[edge] {
                self.sum_diff[edge] += weight as f32 * nwrap as f32;
            } else {
                self.sum_diff[edge] += -weight as f32 * nwrap as f32;
            }
        }
    }

    pub(crate) fn pop_edge(&self) -> (bool, (usize, usize, f32, usize)) {
        let mut max_weight = -1000;
        let mut edge_num = 0usize;
        for (i, &w) in self.weight.iter().enumerate() {
            if w > max_weight {
                max_weight = w;
                edge_num = i;
            }
        }
        if max_weight < 0 {
            return (true, (0, 0, 0.0, 0));
        }
        let node1 = self.node_alpha[edge_num];
        let node2 = self.node_beta[edge_num];
        let diff = self.sum_diff[edge_num] / max_weight as f32;
        (false, (node1, node2, diff, edge_num))
    }
}

fn remove_edge(list: &mut Vec<usize>, edge: usize) {
    if let Some(pos) = list.iter().position(|&e| e == edge) {
        list.remove(pos);
    }
}
