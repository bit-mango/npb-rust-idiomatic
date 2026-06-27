use common::linear_algebra::{euclidean_norm_vec, multiply_vec, scalar_multiply_vec, square_vec};
use common::{LazyRanddp, assert_approx_eq};
use std::collections::HashMap;
use std::usize;

struct CompressedSparseMatrix {
    rows: Vec<SparseRow>,
}

const NONZERO: usize = 7;
const SHIFT: f64 = 10.0; // This is gamma too.

struct SparseRow {
    row_idx: usize,
    non_zero_entries_with_col: Vec<(f64, usize)>,
}

impl SparseRow {
    pub fn new(row_idx: usize) -> Self {
        Self {
            row_idx,
            non_zero_entries_with_col: vec![],
        }
    }

    pub fn add(&mut self, entry: (f64, usize)) {
        self.non_zero_entries_with_col.push(entry);
    }
}

impl CompressedSparseMatrix {
    pub fn new(n: usize) -> Self {
        let mut sparse_vectors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let seed = 314_159_265;
        let a = 5_u64.pow(13);
        // Generate the random numbers we need.
        let mut randdp = LazyRanddp::new(seed, usize::MAX, a);
        // NPB-Rust's main() calls randlc() once before makea() runs, advancing
        // the RNG state by one step. The returned value is discarded (assigned to
        // an unrelated `zeta` variable that gets overwritten later). We replicate
        // this otherwise-undocumented extra draw here to keep our RNG sequence
        // synchronized with the reference implementation.
        randdp.next();
        let ipwr2 = n.next_power_of_two();
        for i in 0..n {
            let mut nz = 0;
            let mut x: HashMap<usize, f64> = HashMap::new();
            while nz < NONZERO {
                let random_num = randdp.next().unwrap();
                let random_idx = (randdp.next().unwrap() * ipwr2 as f64) as usize;

                // If random idx is too large, or already used skip it.
                // println!("random_num: {}, i: {}", random_num, random_idx);
                if random_idx >= n || x.contains_key(&random_idx) {
                    continue;
                }

                x.insert(random_idx, random_num);

                nz += 1;
            }
            // Override ith value of sparse vector to be 0.5.
            x.insert(i, 0.5);
            sparse_vectors.push(x.drain().collect());
        }

        println!("sparse_vectors[0]: {:?}", sparse_vectors[0]);

        // Store results in a hashmap.
        let mut result: HashMap<(usize, usize), f64> = HashMap::new();
        let mut size = 1.0;
        let r = 0.1_f64.powf(1.0 / (n as f64));

        for vec in sparse_vectors.iter() {
            for left in vec {
                for right in vec {
                    // Compute outer product.
                    let outer_product = left.1 * right.1;
                    let key = (left.0, right.0);
                    // Apply weight.
                    let new = size * outer_product;
                    // Check if new already exists.
                    if let Some(existing) = result.get(&key) {
                        result.insert(key, existing + new);
                    } else {
                        result.insert(key, new);
                    }
                }
            }
            size *= r;
        }

        // Add 0.1 and subtract SHIFT to/from the diagonal.
        for i in 0..n {
            if let Some(diagonal_entry) = result.get(&(i, i)) {
                result.insert((i, i), diagonal_entry + 0.1 - SHIFT);
            } else {
                result.insert((i, i), 0.1 - SHIFT);
            }
        }

        let mut intermediate: Vec<((usize, usize), f64)> = result.drain().collect();
        // Sort the intermediate values in ascending order first by row, then by col.
        intermediate.sort_by(|a, b| a.0.0.cmp(&b.0.0).then_with(|| a.0.1.cmp(&b.0.1)));
        let mut rows: Vec<SparseRow> = Vec::new();
        for entry in intermediate.iter() {
            if let Some(row) = rows.last_mut() {
                if row.row_idx == entry.0.0 {
                    row.add((entry.1, entry.0.1));
                } else {
                    let mut new_row = SparseRow::new(entry.0.0);
                    new_row.add((entry.1, entry.0.1));
                    rows.push(new_row);
                }
            } else {
                let mut new_row = SparseRow::new(entry.0.0);
                new_row.add((entry.1, entry.0.1));
                rows.push(new_row);
            }
        }

        Self { rows }
    }

    pub fn multiply(&self, v: &Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; v.len()];

        for row in self.rows.iter() {
            // row_idx is the idx of the result.
            let mut sum = 0.0;
            for entry in row.non_zero_entries_with_col.iter() {
                sum += entry.0 * v[entry.1];
            }
            result[row.row_idx] = sum;
        }

        result
    }
}

fn conjugate_gradient(n: usize, x: &Vec<f64>, a: &mut CompressedSparseMatrix) -> (f64, Vec<f64>) {
    let mut z = vec![0.0; n];
    let mut r = x.clone();
    let mut rho = square_vec(&r);
    let mut p = r.clone();

    for _ in 0..25 {
        let q = a.multiply(&p);
        let alpha = rho / multiply_vec(&p, &q);
        let alpha_p = scalar_multiply_vec(&p, alpha);
        for i in 0..z.len() {
            z[i] += alpha_p[i];
        }
        let rho_not = rho;
        let alpha_q = scalar_multiply_vec(&q, alpha);
        for i in 0..r.len() {
            r[i] -= alpha_q[i];
        }
        rho = square_vec(&r);
        let beta = rho / rho_not;
        let beta_p = scalar_multiply_vec(&p, beta);
        for i in 0..p.len() {
            p[i] = r[i] + beta_p[i];
        }
    }
    let az = a.multiply(&z);
    let mut rn = x.clone();
    for i in 0..rn.len() {
        rn[i] -= az[i];
    }
    (euclidean_norm_vec(&rn), z)
}

fn main() {
    let n = 1_400;
    let gamma = 10.0;
    let mut a = CompressedSparseMatrix::new(n);
    let mut x = vec![1.0_f64; n];
    let niter = 15;
    let mut zeta = 0.0;
    for i in 0..niter {
        let (residual_norm, z) = conjugate_gradient(n, &x, &mut a);
        zeta = gamma + 1.0 / multiply_vec(&x, &z);
        println!(
            "[{}] residual_norm: {:e}, zeta: {:e}",
            i, residual_norm, zeta
        );

        let norm = euclidean_norm_vec(&z);
        for i in 0..x.len() {
            x[i] = z[i] / norm;
        }
    }

    let expected_zeta = 8.59717750786234e0;
    assert_approx_eq(
        zeta,
        expected_zeta,
        1.0e-10,
        "zeta should match EXPECTED_ZETA.",
    );
}
