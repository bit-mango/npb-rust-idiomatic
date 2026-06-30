use common::{Class, LazyRanddp};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::time::{Duration, Instant};

const SEED: u64 = 314_159_265;
const A: u64 = 5_u64.pow(13);

#[derive(Clone, Copy)]
struct IsOutput {
    elapsed_time: Duration,
}
struct IsKernel {
    n: usize,
    b_max: f64,
    i_max: u32,
    output: Option<IsOutput>,
    keys: Vec<i32>,
    test_index_array: Vec<usize>,
    test_rank_array: Vec<u32>,
    add_i: Vec<bool>,
    iter_offset: i32,
}

impl IsKernel {
    pub fn from_class(class: Class) -> Self {
        match class {
            Class::S => Self {
                n: 2_usize.pow(16),
                b_max: 2_f64.powi(11),
                i_max: 9,
                output: None,
                keys: Vec::with_capacity(2_usize.pow(16)),
                test_index_array: vec![48427, 17148, 23627, 62548, 4431],
                test_rank_array: vec![0, 18, 346, 64917, 65463],
                add_i: vec![true, true, true, false, false],
                iter_offset: 0,
            },
            Class::A => Self {
                n: 2_usize.pow(23),
                b_max: 2_f64.powi(19),
                i_max: 10,
                output: None,
                keys: Vec::with_capacity(2_usize.pow(23)),
                test_index_array: vec![2112377, 662041, 5336171, 3642833, 4250760],
                test_rank_array: vec![104, 17523, 123928, 8288932, 8388264],
                add_i: vec![true, true, true, false, false],
                iter_offset: -1,
            },
            Class::B => Self {
                n: 2_usize.pow(25),
                b_max: 2_f64.powi(21),
                i_max: 10,
                output: None,
                keys: Vec::with_capacity(2_usize.pow(25)),
                test_index_array: vec![41869, 812306, 5102857, 18232239, 26860214],
                test_rank_array: vec![33422937, 10244, 59149, 33135281, 99],
                add_i: vec![false, true, true, false, true],
                iter_offset: 0,
            },
            Class::C => Self {
                n: 2_usize.pow(27),
                b_max: 2_f64.powi(23),
                i_max: 10,
                output: None,
                keys: Vec::with_capacity(2_usize.pow(27)),
                test_index_array: vec![44172927, 72999161, 74326391, 129606274, 21736814],
                test_rank_array: vec![61147, 882988, 266290, 133997595, 133525895],
                add_i: vec![true, true, true, false, false],
                iter_offset: 0,
            },
            Class::D => Self {
                n: 2_usize.pow(31),
                b_max: 2_f64.powi(27),
                i_max: 10,
                output: None,
                keys: Vec::with_capacity(2_usize.pow(31)),
                test_index_array: vec![1317351170, 995930646, 1157283250, 1503301535, 1453734525],
                test_rank_array: vec![1, 36538729, 1978098519, 2145192618, 2147425337],
                add_i: vec![true, true, false, false, false],
                iter_offset: 0,
            },
        }
    }

    fn generate_keys(&mut self) {
        // Initialize random number generator.
        let mut lazy_randdp = LazyRanddp::new(SEED, usize::MAX, A);

        for _ in 0..self.n {
            let r_i = vec![
                lazy_randdp.next().unwrap(),
                lazy_randdp.next().unwrap(),
                lazy_randdp.next().unwrap(),
                lazy_randdp.next().unwrap(),
            ];
            let key = (self.b_max * (r_i.iter().sum::<f64>() / 4.0)) as i32;
            self.keys.push(key);
        }
    }

    fn full_verify(&self) {
        for i in 1..self.n {
            assert!(
                self.keys[i - 1] <= self.keys[i],
                "Keys must be sorted in ascending order! keys[{}]: {}, keys[{}]: {}",
                i - 1,
                self.keys[i - 1],
                i,
                self.keys[i]
            );
        }
    }

    #[cfg(feature = "serial")]
    fn compute_ranks(&self) -> Vec<u32> {
        let mut bucket = vec![0_u32; self.b_max as usize];
        for &key in &self.keys {
            bucket[key as usize] += 1;
        }
        // bucket[k] = how many keys have value k.

        for k in 1..bucket.len() {
            bucket[k] += bucket[k - 1];
        }
        // bucket[k] = how many keys have value <= k.
        // so bucket[k-1] = how many keys have value < k = rank of value k.
        bucket
    }

    #[cfg(feature = "parallel")]
    fn compute_ranks(&self) -> Vec<u32> {
        let b_max = self.b_max as usize;

        // Each thread accumulates into its own local bucket, no contention.
        // Then reduce merges by summing element-wise.
        let mut bucket = self
            .keys
            .par_iter()
            .fold(
                || vec![0_u32; b_max],
                |mut local, &key| {
                    local[key as usize] += 1;
                    local
                },
            )
            .reduce(
                || vec![0_u32; b_max],
                |mut a, b| {
                    for i in 0..b_max {
                        a[i] += b[i];
                    }
                    a
                },
            );
        // bucket[k] = how many keys have value k.

        // Sequential prefix sum, each step depends on the previous.
        for k in 1..bucket.len() {
            bucket[k] += bucket[k - 1];
        }
        // bucket[k] = how many keys have value <= k.
        // so bucket[k-1] = how many keys have value < k = rank of value k.
        bucket
    }

    fn partial_verification(&self, ranks: &Vec<u32>, i: u32) {
        for j in 0..5 {
            let key_val = self.keys[self.test_index_array[j]] as usize;
            // bucket[k] = count of keys with value <= k
            // so rank (count < key_val) = bucket[key_val - 1]
            let computed_rank = if key_val == 0 { 0 } else { ranks[key_val - 1] };
            let i_adjusted = (i as i32 + self.iter_offset) as u32;
            let expected = if self.add_i[j] {
                self.test_rank_array[j] + i_adjusted
            } else {
                self.test_rank_array[j] - i_adjusted
            };
            assert_eq!(
                computed_rank, expected,
                "Check failed at i={}, test={}",
                i, j
            );
        }
    }

    fn run(&mut self) {
        let time = Instant::now();
        for i in 1..=self.i_max {
            // 4.a insert test keys
            self.keys[i as usize] = i as i32;
            self.keys[(i + self.i_max) as usize] = (self.b_max as u32 - i) as i32;

            // 4.b compute ranks on modified keys
            let ranks = self.compute_ranks();

            // 4.c partial verification
            self.partial_verification(&ranks, i);
        }
        self.output = Some(IsOutput {
            elapsed_time: time.elapsed(),
        });
    }

    fn debug_print(&self) {
        let output = self
            .output
            .expect("run() must be called before debug_print()");

        let total_time_us = output.elapsed_time.as_micros();
        println!("Completed in {:.3} seconds", total_time_us as f64 / 1.0e6);

        let mops = (self.i_max as usize * self.n) as u128 / output.elapsed_time.as_micros();
        println!("MOPs: {}", mops);
    }

    fn sort_keys(&mut self) {
        let bucket = self.compute_ranks();
        let mut sorted = Vec::with_capacity(self.n);
        let mut prev = 0_u32;
        for (k, &cumulative) in bucket.iter().enumerate() {
            let freq = cumulative - prev;
            for _ in 0..freq {
                sorted.push(k as i32);
            }
            prev = cumulative;
        }
        self.keys = sorted;
    }
}
fn main() {
    let class = Class::C;
    let mut kernel = IsKernel::from_class(class);
    kernel.generate_keys();
    kernel.run();
    kernel.sort_keys();
    kernel.full_verify();
    kernel.debug_print();
}
