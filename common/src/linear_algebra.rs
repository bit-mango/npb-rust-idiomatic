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
