use std::time::Duration;

pub mod linear_algebra;
// Parallelism
// x_(k+1) = a * x_k mod 2^46
// x_0 = seed
// x_1 = a * x_0 mod 2^46 = a * seed mod 2^46
// x_2 = a * x_1 mod 2^46 = a * (a * seed mod 2^46) mod 2^46
// x_3 = a * x_2 mod 2^46 = a * (a * (a * seed mod 2^46) mod 2^46) mod 2^46
// x_4 = a * x_3 mod 2^46 = a * (a * (a * (a * seed mod 2^46) mod 2^46) mod 2^46) mod 2^46) mod 2^46
// x_k = a^(k-1) * seed mod 2^46
// Then we take k, say its 13, which in binary is 1101. a^13 = a^1 * a^4 * a^8
// Take a mod 2^46 after every multiplication, so its (((a^1 mod 2^46) * a^4 mod 2^46) * a^8 mod 2^46)

const B: u64 = 1_u64 << 46;
const C: f64 = 1.0 / (1u64 << 46) as f64; // 2.0_f64.powi(-46)
// Want to generate n uniform psuedo random numbers.
// a = 5^13
// x_0 = s, a specified initial seed. Where 0 < s < 2^46
// Generate the integers x_k for 1 <= k <= n using the following:
// x_k+1 = a*x_k % 2^46
// Then return r_k = 2^(-46)*x_k
// Thus 0 < r_k < 1

// Before LazyRanddp size was Option<u128> + usize + u128
// or 1 + 16 + 8 + 16 = 40
// After optimization storing u64 instead
// 24
#[derive(Clone, Copy)]
pub struct LazyRanddp {
    current: u64,
    n: usize,
    a: u64,
}

impl LazyRanddp {
    pub fn new(seed: u64, n: usize, a: u64) -> Self {
        Self {
            current: seed,
            n,
            a,
        }
    }

    pub fn get_current(&self) -> u64 {
        self.current
    }

    // x_k = a^k * seed mod 2^46
    // Then we take k, say its 13, which in binary is 1101. a^13 = a^1 * a^4 * a^8
    // Take a mod 2^46 after every multiplication, so its (((a^1 mod 2^46) * a^4 mod 2^46) * a^8 mod 2^46)
    pub fn skip_forward(&mut self, amount: u64) {
        if amount >= u32::MAX as u64 {
            panic!("TOO BIG!");
        }
        // Perform bitwise exponentiation.
        let mut mask = 1_u64;
        let mut scaler = self.current;
        while mask <= amount {
            let exp = amount & mask;
            scaler = scaler.wrapping_mul(self.a.wrapping_pow(exp as u32));
            mask = mask << 1;
        }

        self.current = scaler % B;
    }

    pub fn to_vec(self) -> Vec<f64> {
        let mut r_k = Vec::with_capacity(self.n);
        for elem in self {
            r_k.push(elem);
        }

        r_k
    }
}

impl Iterator for LazyRanddp {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            return None;
        } else {
            self.n -= 1;
        }
        // Compute the next value.
        // Use wrapping_mul as its equivalent to ((a * current) mod 2^64) mod 2^46.
        let next = self.a.wrapping_mul(self.current) % B;

        // Update current.
        self.current = next;

        // Compute r.
        Some(C * next as f64)
    }
}

pub struct ParallelLazyRanddp {
    current: u64,
    prev_n: usize,
    a: u64,
    m: usize,
    batch_size: usize,
    remainder: usize,
}
impl ParallelLazyRanddp {
    pub fn new(seed: u64, n: usize, a: u64, m: usize) -> Self {
        if m % 2 != 0 {
            panic!("Only even threads.");
        }
        let mut batch_size = n / m; // batch size for all iterators.
        let mut remainder = 0; // Remainder given to first iterator.
        if batch_size % 2 != 0 {
            // Need to lower batch size to make it even.
            batch_size -= 1;
            // Add to remainder.
            remainder += m; // Every thread had 1 that needed to be
        }

        // Add remainder from batching.
        remainder += n % m;

        if batch_size % 2 != 0 {
            panic!("Batch size is odd!");
        }

        if remainder % 2 != 0 {
            panic!("remainder is odd!");
        }

        Self {
            current: seed,
            prev_n: 0,
            a,
            m,
            batch_size,
            remainder,
        }
    }

    pub fn to_vec(self) -> Vec<LazyRanddp> {
        let mut iters = Vec::with_capacity(self.m);
        for elem in self {
            iters.push(elem);
        }

        iters
    }
}

impl Iterator for ParallelLazyRanddp {
    type Item = LazyRanddp;

    fn next(&mut self) -> Option<Self::Item> {
        if self.m == 0 {
            return None;
        } else {
            self.m -= 1;
        }
        let n = if self.prev_n == 0 {
            // First iterator, give it the remainder
            self.batch_size + self.remainder
        } else {
            self.batch_size
        };

        let mut lazy_randdp = LazyRanddp::new(self.current, n, self.a);
        // Skip lazy_randdp forward.
        if self.prev_n >= u64::MAX as usize {
            panic!("Oh no!");
        }
        lazy_randdp.skip_forward(self.prev_n as u64);
        // Update current, so this calculation doesn't need to be repeated.
        self.current = lazy_randdp.current;
        // Update prev_n.
        self.prev_n = n;

        Some(lazy_randdp)
    }
}

pub fn assert_approx_eq(left: f64, right: f64, epsilon: f64, explanation: &str) {
    let relative_error = (left - right).abs() / right.abs();
    assert!(
        relative_error < epsilon,
        "{}: relative_error: {}, epsilon: {}, left: {}, right: {}",
        explanation,
        relative_error,
        epsilon,
        left,
        right
    );
}

pub enum Class {
    S, // Sample
    A,
    B,
    C,
    D,
}

pub fn print_results(
    _elapsed: Duration,
    _kernel_name: &str,
    _class: Class,
    _problem_size: usize,
    _operation_count: u64,
) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn validate_rngddp_range() {
        // All random numbers, r, should follow this rule 0 < r < 1.
        let npb_seed = 271828182845904523;
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let r = lazy_randdp.to_vec();
        for elem in r.iter() {
            assert!(
                *elem > 0.0 && *elem < 1.0,
                "Random numbers should fall between 0 and 1, got {:}",
                elem
            );
        }
    }

    #[test]
    fn validate_rngddp_distribution() {
        // Should be evenly distributed such that n/10 numbers fall between 0.0 -> 0.1 and so on.
        let npb_seed = 271828182845904523;
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let r = lazy_randdp.to_vec();
        let mut ranges = vec![];
        let mut lower = 0.0;
        let mut upper = 0.1;
        for _ in 0..10 {
            ranges.push((lower, upper));
            lower = upper;
            upper += 0.1;
        }

        let mut count = vec![0_u32; 10];
        for elem in r.iter() {
            for j in 0..10 {
                if ranges[j].0 < *elem && *elem <= ranges[j].1 {
                    count[j] = count[j] + 1;
                    break;
                }
            }
        }

        for i in 0..10 {
            assert!(
                count[i] > 90 && count[i] < 110,
                "Expected equal distribution, but got {}, in {}",
                count[i],
                i
            );
        }
    }

    #[test]
    fn validate_lazy_rngddp_range() {
        // All random numbers, r, should follow this rule 0 < r < 1.
        let npb_seed = 271828182845904523;
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let mut count = 0;
        for elem in lazy_randdp {
            count += 1;
            assert!(
                elem > 0.0 && elem < 1.0,
                "Random numbers should fall between 0 and 1, got {:}",
                elem
            );
        }
        assert_eq!(count, 1_000, "Should have generated 1_000 random numbers.");
    }

    #[test]
    fn validate_lazy_rngddp_distribution() {
        // Should be evenly distributed such that n/10 numbers fall between 0.0 -> 0.1 and so on.
        let npb_seed = 271828182845904523;
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let mut ranges = vec![];
        let mut lower = 0.0;
        let mut upper = 0.1;
        for _ in 0..10 {
            ranges.push((lower, upper));
            lower = upper;
            upper += 0.1;
        }

        let mut count = vec![0_u32; 10];
        for elem in lazy_randdp {
            for j in 0..10 {
                if ranges[j].0 < elem && elem <= ranges[j].1 {
                    count[j] = count[j] + 1;
                    break;
                }
            }
        }

        for i in 0..10 {
            assert!(
                count[i] > 90 && count[i] < 110,
                "Expected equal distribution, but got {}, in {}",
                count[i],
                i
            );
        }
    }

    #[test]
    fn test_skip_forward() {
        // Generate 1,000 psuedo random numbers.
        let npb_seed = 271828182845904523;
        let lazy_randdp = LazyRanddp::new(npb_seed, 10_000, 5_u64.pow(13));
        let r = lazy_randdp.to_vec();

        let mut lazy_randdp = LazyRanddp::new(npb_seed, 10, 5_u64.pow(13));

        // Check some numbers.
        let indexes_to_check = [16_u32, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        let mut last_index_generated = 0;
        for index in indexes_to_check {
            let amount_to_skip = index - last_index_generated;
            lazy_randdp.skip_forward(amount_to_skip as u64);
            let skip_forward_generated = lazy_randdp.next().unwrap();
            assert_eq!(
                r[index as usize], skip_forward_generated,
                "Numbers should match, failed at index: {}",
                index
            );
            last_index_generated = index + 1;
        }
    }

    #[test]
    fn test_parallel_lazy_randdp_no_rayon() {
        let npb_seed = 271828182845904523;
        let parallel_lazy_randdp = ParallelLazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13), 7);

        let iters = parallel_lazy_randdp.to_vec();
        let mut r_k: Vec<f64> = vec![];
        for iter in iters {
            let mut r = iter.to_vec();
            r_k.append(&mut r);
        }

        // Generate it normally
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let original = lazy_randdp.to_vec();

        assert_eq!(
            r_k, original,
            "Parallel generated vec does not match original."
        );
    }

    #[test]
    fn test_parallel_debug() {
        let npb_seed = 271828182845904523;
        let n = 2 << 28;
        let parallel_lazy_randdp = ParallelLazyRanddp::new(npb_seed, n, 5_u64.pow(13), 10);
        let mut lazy_randdp = LazyRanddp::new(npb_seed, n, 5_u64.pow(13));

        let mut iters = parallel_lazy_randdp.to_vec();
        while let (Some(a), Some(b)) = (iters[0].next(), lazy_randdp.next()) {
            assert_eq!(a, b, "Should be the same");
        }
    }

    #[test]
    fn test_parallel_lazy_randdp_with_rayon() {
        let npb_seed = 271828182845904523;
        let parallel_lazy_randdp = ParallelLazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13), 7);

        let iters = parallel_lazy_randdp.to_vec();
        let rngs: Vec<Vec<f64>> = iters
            .par_iter()
            .map(|lazy_randdp| lazy_randdp.to_vec())
            .collect();
        let mut r_k: Vec<f64> = vec![];
        for mut rng in rngs {
            r_k.append(&mut rng);
        }

        // Generate it normally
        let lazy_randdp = LazyRanddp::new(npb_seed, 1_000, 5_u64.pow(13));
        let original = lazy_randdp.to_vec();

        assert_eq!(
            r_k, original,
            "Parallel generated vec does not match original."
        );
    }
}
