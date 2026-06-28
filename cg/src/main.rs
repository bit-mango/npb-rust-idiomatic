use common::{Class, LazyRanddp, assert_approx_eq};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::usize;

const SEED: u64 = 314_159_265;
const A: u64 = 5_u64.pow(13);

// #[cfg(class = "S")]
// mod params {
//     const N: usize = 1400;
//     const N_ITER: usize = 15;
//     const NONZERO: usize = 7;
//     const GAMMA: f64 = 10.0;
// }
//
#[derive(Clone, Copy)]
struct CgOutput {
    elapsed_time: Duration,
    final_zeta: f64,
}

struct CgKernel {
    n: usize,
    n_iter: usize,
    nonzero: usize,
    gamma: f64,
    expected_zeta: f64,
    output: Option<CgOutput>,
}

impl CgKernel {
    pub fn from_class(class: Class) -> Self {
        match class {
            Class::S => Self {
                n: 1_400,
                n_iter: 15,
                nonzero: 7,
                gamma: 10.0,
                expected_zeta: 8.59717750786234,
                output: None,
            },
            Class::A => Self {
                n: 14_000,
                n_iter: 15,
                nonzero: 11,
                gamma: 20.0,
                expected_zeta: 17.13023505380784,
                output: None,
            },
            Class::B => Self {
                n: 75_000,
                n_iter: 75,
                nonzero: 13,
                gamma: 60.0,
                expected_zeta: 22.712745482631,
                output: None,
            },
            Class::C => Self {
                n: 150_000,
                n_iter: 75,
                nonzero: 15,
                gamma: 110.0,
                expected_zeta: 28.973605592845,
                output: None,
            },
            Class::D => Self {
                n: 1_500_000,
                n_iter: 100,
                nonzero: 21,
                gamma: 500.0,
                expected_zeta: 52.514532105794,
                output: None,
            },
        }
    }

    pub fn verify(&self) {
        let output = self.output.expect("run() must be called before verify()");

        assert_approx_eq(
            output.final_zeta,
            self.expected_zeta,
            1.0e-10,
            "zeta should match EXPECTED_ZETA.",
        );
    }

    pub fn run(&mut self) {
        let mut a = CompressedSparseMatrix::new(self.n, self.gamma, self.nonzero);
        let mut x = vec![1.0_f64; self.n];
        let mut zeta = 0.0;
        let time = Instant::now();

        for i in 0..self.n_iter {
            let (residual_norm, z) = conjugate_gradient(self.n, &x, &mut a);
            zeta = self.gamma + 1.0 / multiply_vec(&x, &z);
            println!(
                "[{}] residual_norm: {:e}, zeta: {:e}",
                i, residual_norm, zeta
            );

            let norm = euclidean_norm_vec(&z);
            for i in 0..self.n {
                x[i] = z[i] / norm;
            }
        }

        self.output = Some(CgOutput {
            elapsed_time: time.elapsed(),
            final_zeta: zeta,
        })
    }

    pub fn print(&self) {
        let output = self.output.expect("run() must be called before print()");

        let time_elapsed = output.elapsed_time;
        let total_time_us = time_elapsed.as_micros();
        println!("Completed in {:.3} seconds", total_time_us as f64 / 1.0e6);
    }
}

struct CompressedSparseMatrix {
    rows: Vec<(usize, usize, usize)>, // row_idx, start_idx, end_idx
    data: Vec<(usize, f64)>,          // col_idx, value
}

impl CompressedSparseMatrix {
    pub fn new(n: usize, shift: f64, nonzero: usize) -> Self {
        let mut sparse_vectors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        // Generate the random numbers we need.
        // Use usize::MAX because the exact amount of numbers is not known until it is run as some are thrown away.
        let mut randdp = LazyRanddp::new(SEED, usize::MAX, A);
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
            while nz < nonzero {
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

        // Add 0.1 and subtract shift to/from the diagonal.
        for i in 0..n {
            if let Some(diagonal_entry) = result.get(&(i, i)) {
                result.insert((i, i), diagonal_entry + 0.1 - shift);
            } else {
                result.insert((i, i), 0.1 - shift);
            }
        }

        let mut intermediate: Vec<((usize, usize), f64)> = result.drain().collect();
        // Sort the intermediate values in ascending order first by row, then by col.
        intermediate.sort_by(|a, b| a.0.0.cmp(&b.0.0).then_with(|| a.0.1.cmp(&b.0.1)));
        let mut s = Self {
            rows: vec![],
            data: vec![],
        };
        let mut start_data_idx = 0;
        let mut last_row_idx = intermediate[start_data_idx].0.0;
        for (i, entry) in intermediate.iter().enumerate() {
            if entry.0.0 != last_row_idx {
                // On a new row now.
                s.rows.push((last_row_idx, start_data_idx, i));
                last_row_idx = entry.0.0;
                start_data_idx = i;
            }

            // Add entry to data.
            s.data.push((entry.0.1, entry.1));
        }

        // Add final row.
        s.rows.push((last_row_idx, start_data_idx, s.data.len()));

        s
    }

    pub fn multiply(&self, v: &Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; v.len()];

        for row in self.rows.iter() {
            // row_idx is the idx of the result.
            let mut sum = 0.0;
            for r in &self.data[row.1..row.2] {
                sum += v[r.0] * r.1;
            }
            result[row.0] = sum;
        }

        result
    }
}

pub fn square_vec(v: &Vec<f64>) -> f64 {
    v.iter().map(|e| e * e).sum()
}

pub fn multiply_vec(l: &Vec<f64>, r: &Vec<f64>) -> f64 {
    l.iter().zip(r.iter()).map(|e| e.0 * e.1).sum()
}

pub fn euclidean_norm_vec(v: &Vec<f64>) -> f64 {
    square_vec(v).sqrt()
}

pub fn scalar_multiply_vec(v: &Vec<f64>, s: f64) -> Vec<f64> {
    v.iter().map(|e| e * s).collect()
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
        for i in 0..n {
            z[i] += alpha_p[i];
        }
        let rho_not = rho;
        let alpha_q = scalar_multiply_vec(&q, alpha);
        for i in 0..n {
            r[i] -= alpha_q[i];
        }
        rho = square_vec(&r);
        let beta = rho / rho_not;
        let beta_p = scalar_multiply_vec(&p, beta);
        for i in 0..n {
            p[i] = r[i] + beta_p[i];
        }
    }
    let az = a.multiply(&z);
    let mut rn = x.clone();
    for i in 0..n {
        rn[i] -= az[i];
    }
    (euclidean_norm_vec(&rn), z)
}

fn main() {
    let mut kernel = CgKernel::from_class(Class::B);
    kernel.run();
    kernel.verify();
    kernel.print();
}
