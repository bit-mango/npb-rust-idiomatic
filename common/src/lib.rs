use std::time::Duration;
// randdp, timing, verification, class sizes

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
// TODO create skip_to function, then make a special function that returns an iterator of iterators.

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
        let next = ((self.a as u128 * self.current as u128) % B as u128) as u64;

        // Update current.
        self.current = next;

        // Compute r.
        Some(C * next as f64)
    }
}

pub fn randdp(seed: u64, n: usize, a: u64) -> Vec<f64> {
    let mut r_k = Vec::with_capacity(n);
    let lazy_randdp = LazyRanddp::new(seed, n, a);
    for elem in lazy_randdp {
        r_k.push(elem);
    }

    r_k
}

pub fn assert_approx_eq(left: f64, right: f64, epsilon: f64, explanation: &str) {
    let relative_error = (left - right).abs() / right.abs();
    assert!(
        relative_error < epsilon,
        "{}: relative_error: {}, epsilon: {}",
        explanation,
        relative_error,
        epsilon
    );
}

pub enum Class {
    S, // Sample
    A,
    B,
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

    #[test]
    fn validate_rngddp_range() {
        // All random numbers, r, should follow this rule 0 < r < 1.
        let npb_seed = 271828182845904523;
        let r = randdp(npb_seed, 1_000, 5_u64.pow(13));
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
        let r = randdp(npb_seed, 1_000, 5_u64.pow(13));
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
}
