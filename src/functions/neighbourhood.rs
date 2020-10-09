/// Selection of neighborhood function implementations
///

use ndarray;
use ndarray::{Array2};
use std::f64::consts::PI as PI;

// Default neighbourhood function: Gaussian function; returns a Gaussian centered in pos
pub fn gaussian(size: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {
    let mut ret = Array2::<f64>::zeros((size.0, size.1));
    let div = 2.0 * PI * sigma as f64 * sigma as f64;

    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for i in 0..size.0 {
        x.push(i as f64);
        if let Some(elem) = x.get_mut(i) {
            *elem = -((*elem - (pos.0 as f64)).powf(2.0) / div);
            *elem = (*elem).exp();
        }
    }

    for i in 0..size.1 {
        y.push(i as f64);
        if let Some(elem) = y.get_mut(i) {
            *elem = -((*elem - (pos.1 as f64)).powf(2.0) / div);
            *elem = (*elem).exp();
        }
    }

    for i in 0..size.0 {
        for j in 0..size.1 {
            ret[[i, j]] = x[i] * y[j];
        }
    }

    ret
}

//pub fn mexican_hat(size: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {

//}
