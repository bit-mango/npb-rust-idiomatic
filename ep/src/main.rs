#[cfg(feature = "lazy")]
use common::LazyRanddp;
#[cfg(feature = "eager")]
use common::randdp;
use common::{Class, assert_approx_eq};
use std::time::{Duration, Instant};

// TODO Add classes
// TODO Add print_results to common.
// TODO optimize
// Thinking this could maybe be a struct with generics that are set using cfg flags at compile time?
//
// TODO if I need feature specific compilation for array lengths.
// #[cfg(feature = "S")]
// const N: usize = 1 << 24;

// #[cfg(feature = "A")]
// const N: usize = 1 << 28;

// #[cfg(feature = "B")]
// const N: usize = 1 << 28;

// TODO could use inlining with functions for criterion benchmarking once I have iterators in place.
// // A basic inline hint (allows cross-crate optimization)
// #[inline]
// pub fn add_small(a: i32, b: i32) -> i32 {
//     a + b
// }

// // Forcing the compiler's hand for performance-critical hotspots
// #[inline(always)]
// pub fn tight_loop_utility(x: f64) -> f64 {
//     x.abs() * 2.0
// }

// // Preventing inlining on a heavy, rarely executed error path
// #[inline(never)]
// pub fn log_critical_error_and_panic(msg: &str) -> ! {
//     println!("CRITICAL: {}", msg);
//     panic!("Execution halted");
// }

#[derive(Clone, Copy)]
struct EpOutput {
    elapsed_time: Duration,
    counts: [u64; 10],
    sum_x: f64,
    sum_y: f64,
}
struct EpKernel {
    n: usize,
    a: u64,
    s: u64,
    expected_x_k_sum: f64,
    expected_y_k_sum: f64,
    expected_counts: [u64; 10],
    m_flops: f64,
    output: Option<EpOutput>,
}

// const kernel: EpKernel = EpKernel::from_class(Class::A); // Doesnt work unless you make from_class const

impl EpKernel {
    pub fn from_class(class: Class) -> Self {
        match class {
            Class::S => Self {
                n: 1 << 24,
                a: 5_u64.pow(13),
                s: 271_828_183,
                expected_x_k_sum: -3.2478346520347404e3,
                expected_y_k_sum: -6.958407078382297e3,
                expected_counts: [6140517, 5865300, 1100361, 68546, 1648, 17, 0, 0, 0, 0],
                m_flops: 1.392e3,
                output: None,
            },
            Class::A => Self {
                n: 1 << 28,
                a: 5_u64.pow(13),
                s: 271_828_183,
                expected_x_k_sum: -4.295875165629892e3,
                expected_y_k_sum: -1.580732573678431e4,
                expected_counts: [
                    98257395, 93827014, 17611549, 1110028, 26536, 245, 0, 0, 0, 0,
                ],
                m_flops: 22.197e3,
                output: None,
            },
            Class::B => Self {
                n: 1 << 30,
                a: 5_u64.pow(13),
                s: 271_828_183,
                expected_x_k_sum: 4.033815542441498e4,
                expected_y_k_sum: -2.660669192809235e4,
                expected_counts: [
                    393058470, 375280898, 70460742, 4438852, 105691, 948, 5, 0, 0, 0,
                ],
                m_flops: 100.864e3,
                output: None,
            },
        }
    }

    pub fn verify(&self) {
        let output = self.output.expect("run() must be called before verify()");

        assert_eq!(
            output.counts, self.expected_counts,
            "Counts should match EXPECTED_COUNTS."
        );
        assert_approx_eq(
            output.sum_x,
            self.expected_x_k_sum,
            1.0e-12,
            "sum_x should match EXPECTED_X_K_SUM.",
        );
        assert_approx_eq(
            output.sum_y,
            self.expected_y_k_sum,
            1.0e-12,
            "sum_y should match EXPECTED_Y_K_SUM.",
        );
    }

    // Runs in 3.96 sec for Class A.
    #[cfg(feature = "eager")]
    pub fn run(&mut self) {
        let time = Instant::now();
        // TODO can randdp stream random numbers one at a time instead of eagerly building a massive vec of random numbers.
        let r = randdp(self.s, 2 * self.n, self.a);
        let mut x = vec![0.0_f64; self.n];
        let mut y = vec![0.0_f64; self.n];
        // Repeated math here might be able to do something where we set x[i] = y[i-1] maybe? Just need to handle first iteration.
        for i in 0..self.n {
            x[i] = 2.0 * r[2 * i] - 1.0;
            y[i] = 2.0 * r[2 * i + 1] - 1.0;
        }

        let mut k = 0;
        let mut y_k = vec![];
        let mut x_k = vec![];

        for i in 0..self.n {
            let t = x[i].powi(2) + y[i].powi(2);
            if t <= 1.0 {
                x_k.push(x[i] * (-2.0 * t.ln() / t).sqrt());
                y_k.push(y[i] * (-2.0 * t.ln() / t).sqrt());
                k += 1;
            }
        }

        let mut counts = [0_u64; 10];
        // This seems to be about 10% faster but save this for actual optimization once benchmarking is set up.
        // for i in 0..k {
        //     let max_abs = x_k[i].abs().max(y_k[i].abs());
        //     counts[max_abs as usize] += 1;
        // }
        for i in 0..k {
            let max = x_k[i].abs().max(y_k[i].abs());
            // Figure out what bin it belongs too.
            for l in 0..10 {
                if l as f64 <= max && max < (l + 1) as f64 {
                    // Found the bin.
                    counts[l] += 1;
                    break;
                }
            }
        }

        let sum_x: f64 = x_k.iter().sum();
        let sum_y: f64 = y_k.iter().sum();
        self.output = Some(EpOutput {
            elapsed_time: time.elapsed(),
            counts,
            sum_x,
            sum_y,
        });
    }

    // Runs in 2.3 sec for Class A.
    #[cfg(feature = "lazy")]
    pub fn run(&mut self) {
        let time = Instant::now();
        let mut lazy_randdp = LazyRanddp::new(self.s, 2 * self.n, self.a);

        let mut counts = [0_u64; 10];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        while let (Some(a), Some(b)) = (lazy_randdp.next(), lazy_randdp.next()) {
            let x = 2.0 * a - 1.0;
            let y = 2.0 * b - 1.0;
            let t = x * x + y * y;
            if t <= 1.0 {
                let scale = (-2.0 * t.ln() / t).sqrt();
                let x_k = x * scale;
                let y_k = y * scale;
                let max_abs = x_k.abs().max(y_k.abs());
                counts[max_abs as usize] += 1;
                sum_x += x_k;
                sum_y += y_k;
            }
        }

        self.output = Some(EpOutput {
            elapsed_time: time.elapsed(),
            counts,
            sum_x,
            sum_y,
        });
    }

    pub fn debug_print(&self) {
        let output = self
            .output
            .expect("run() must be called before debug_print()");

        let total_time_us = output.elapsed_time.as_micros();
        println!("Completed in {:.3} seconds", total_time_us as f64 / 1.0e6);
        println!("Mflops/s: {:.0}", self.m_flops * 1e6 / total_time_us as f64);
        println!("sum_x: {:.10e}, sum_y: {:.10e}", output.sum_x, output.sum_y);
        for l in 0..10 {
            println!("l[{}]: {}", l, output.counts[l]);
        }
    }
}

fn main() {
    let class = Class::B;
    let mut kernel = EpKernel::from_class(class);
    kernel.run();
    kernel.verify();
    kernel.debug_print();
}
