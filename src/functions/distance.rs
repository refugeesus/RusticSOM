/// Node distance calculation functions
///
///

use ndarray;
use ndarray::ArrayView1;

// Returns the euclidian distance between 2 vectors
pub fn euclid_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Both arrays must be of same length to find Euclidian distance!");
    }

    let mut dist: f64 = 0.0;

    for i in 0..a.len() {
        dist += (a[i] - b[i]).powf(2.0);
    }

    dist.powf(0.5)
}
