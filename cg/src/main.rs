use common::{Class, LazyRanddp, assert_approx_eq};
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

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
        let mut sp = ScratchPad {
            z: vec![0.0; self.n],
            r: vec![0.0; self.n],
            p: vec![0.0; self.n],
            q: vec![0.0; self.n],
            az: vec![0.0; self.n],
            rn: vec![0.0; self.n],
        };
        let time = Instant::now();

        for i in 0..self.n_iter {
            let residual_norm = conjugate_gradient(self.n, &x, &mut a, &mut sp);
            zeta = self.gamma + 1.0 / multiply_vec(&x, &sp.z);
            println!(
                "[{}] residual_norm: {:e}, zeta: {:e}",
                i, residual_norm, zeta
            );

            let norm = euclidean_norm_vec(&sp.z);
            for i in 0..self.n {
                x[i] = sp.z[i] / norm;
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
    rows: Vec<u32>, // start_idx. No need to store row_idx because rows are sequential and we are guaranteed to have atleast
    // one non zero element in each row because of the diagonal shift.
    // TODO The size of this tuple in memory is 16 bytes, so possible optimization is to store this as 12 bytes
    data: Vec<(u32, f64)>, // col_idx, value
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
        // Add dummy entry so that final row is pushed.
        intermediate.push(((usize::MAX, 0), 0.0));
        let mut s = Self {
            rows: vec![],
            data: vec![],
        };
        let mut start_data_idx = 0;
        let mut last_row_idx = 0;
        for (i, entry) in intermediate.iter().enumerate() {
            if entry.0.0 != last_row_idx {
                // On a new row now.
                s.rows.push(start_data_idx as u32);
                last_row_idx += 1;
                start_data_idx = i;
            }

            // Add entry to data.
            s.data.push((entry.0.1 as u32, entry.1));
        }

        // Add n+1 row entry so final row knows when to stop.
        s.rows.push(s.data.len() as u32);

        s
    }

    #[cfg(feature = "serial")]
    pub fn multiply(&self, v: &Vec<f64>, result: &mut Vec<f64>) {
        // The start of the next row is the end of the current row, so zip them into iterators.
        (&self.rows[0..v.len()])
            .into_iter()
            .zip(&self.rows[1..v.len() + 1])
            .zip(result)
            .for_each(|((start, stop), res)| {
                let mut sum = 0.0;
                for r in &self.data[*start as usize..*stop as usize] {
                    sum += v[r.0 as usize] * r.1;
                }
                *res = sum;
            });
    }

    #[cfg(feature = "parallel")]
    pub fn multiply(&self, v: &Vec<f64>, result: &mut Vec<f64>) {
        // The start of the next row is the end of the current row, so zip them into iterators.
        (&self.rows[0..v.len()], &self.rows[1..v.len() + 1])
            .into_par_iter()
            .zip(result.par_iter_mut())
            .for_each(|((start, stop), res)| {
                let mut sum = 0.0;
                for r in &self.data[*start as usize..*stop as usize] {
                    sum += v[r.0 as usize] * r.1;
                }
                *res = sum;
            });
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

pub fn scale_and_add(v: &Vec<f64>, scalar: f64, result: &mut Vec<f64>) {
    for i in 0..result.len() {
        result[i] = v[i] + scalar * result[i];
    }
}

pub fn scale_and_increment(v: &Vec<f64>, scalar: f64, result: &mut Vec<f64>) {
    for i in 0..result.len() {
        result[i] += scalar * v[i];
    }
}

pub fn scale_and_decrement(v: &Vec<f64>, scalar: f64, result: &mut Vec<f64>) {
    for i in 0..result.len() {
        result[i] -= scalar * v[i];
    }
}

struct ScratchPad {
    pub z: Vec<f64>,
    pub r: Vec<f64>,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    pub az: Vec<f64>,
    pub rn: Vec<f64>,
}

fn conjugate_gradient(
    n: usize,
    x: &Vec<f64>,
    a: &mut CompressedSparseMatrix,
    sp: &mut ScratchPad,
) -> f64 {
    sp.z.fill(0.0);
    sp.r.copy_from_slice(x);
    let mut rho = square_vec(&sp.r);
    sp.p.copy_from_slice(x);

    for _ in 0..25 {
        a.multiply(&sp.p, &mut sp.q);
        let alpha = rho / multiply_vec(&sp.p, &sp.q);
        scale_and_increment(&sp.p, alpha, &mut sp.z);
        let rho_not = rho;
        scale_and_decrement(&sp.q, alpha, &mut sp.r);
        rho = square_vec(&sp.r);
        let beta = rho / rho_not;
        scale_and_add(&sp.r, beta, &mut sp.p);
    }
    a.multiply(&sp.z, &mut sp.az);
    sp.rn.copy_from_slice(x);
    for i in 0..n {
        sp.rn[i] -= sp.az[i];
    }
    euclidean_norm_vec(&sp.rn)
}

// type Item = (u32, u32, u32);

fn main() {
    // println!("size of Item: {}", size_of::<Item>());
    let mut kernel = CgKernel::from_class(Class::C);
    kernel.run();
    kernel.verify();
    kernel.print();
}
