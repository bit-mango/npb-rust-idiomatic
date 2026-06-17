use common::{assert_approx_eq, randdp};
use std::time::Instant;

const N: usize = 1 << 28; // 2^28
const EXPECTED_COUNTS: [u64; 10] = [
    98257395, 93827014, 17611549, 1110028, 26536, 245, 0, 0, 0, 0,
];
const EXPECTED_X_K_SUM: f64 = -4.295875165629892e3;
const EXPECTED_Y_K_SUM: f64 = -1.580732573678431e4;

fn main() {
    let time = Instant::now();
    let a = 5_u128.pow(13);
    let s = 271_828_183;
    let r = randdp(s, 2 * N, a);
    let mut x = vec![0.0_f64; N];
    let mut y = vec![0.0_f64; N];
    for i in 0..N {
        x[i] = 2.0 * r[2 * i] - 1.0;
        y[i] = 2.0 * r[2 * i + 1] - 1.0;
    }

    let mut k = 0;
    let mut y_k = vec![];
    let mut x_k = vec![];

    for i in 0..N {
        let t = x[i].powi(2) + y[i].powi(2);
        if t <= 1.0 {
            x_k.push(x[i] * (-2.0 * t.ln() / t).sqrt());
            y_k.push(y[i] * (-2.0 * t.ln() / t).sqrt());
            k += 1;
        }
    }

    let mut counts = [0_u64; 10];
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
    println!("Completed in {} seconds", time.elapsed().as_secs());
    println!("sum_x: {:.10e}, sum_y: {:.10e}", sum_x, sum_y);

    assert_eq!(
        counts, EXPECTED_COUNTS,
        "Counts should match EXPECTED_COUNTS."
    );
    assert_approx_eq(
        sum_x,
        EXPECTED_X_K_SUM,
        1.0e-12,
        "sum_x should match EXPECTED_X_K_SUM.",
    );
    assert_approx_eq(
        sum_y,
        EXPECTED_Y_K_SUM,
        1.0e-12,
        "sum_y should match EXPECTED_Y_K_SUM.",
    );
    for l in 0..10 {
        println!("l[{}]: {}", l, counts[l]);
    }
}
