/// Selection of decay functions
///
///
pub fn exponential_decay(val: f64, iter: u32, max_iter: u32) -> f64 {
    println!("{:?}", ((- (iter as f64)/max_iter as f64)).exp());
    let ex = ((- (iter as f64)/max_iter as f64)).exp();
    val*ex
}

pub fn power_decay(start: f64, end: f64, iter: u32, max_iter: u32) -> f64 {
    let p = (iter as f64 / max_iter as f64) as f64;
    let lr = start * (start/end);
    lr.powf(p)
}
