use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::random;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Classification {
    label: String,
    weight: f64,
    global_weight: f64,
}

impl Classification {
    pub fn new(label: String, weight: f64) -> Classification {
        Classification {
            label,
            weight,
            global_weight: 0.0,
        }
    }
}

impl Default for Classification {
    fn default() -> Self {
        Classification {
            label: "undefined".to_string(),
            weight: 0.0,
            global_weight: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SomData {
    x: usize,                                   // length of SOM
    y: usize,                                   // breadth of SOM
    z: usize,                                   // size of inputs
    learning_rate: f32,                         // initial learning rate
    sigma: f32,                                 // spread of neighbourhood function, default = 1.0
    regulate_lrate: u32, // Regulates the learning rate w.r.t the number of iterations
    maximum_iterations: u32, // Maximum number of iterations per training session
    pub map: Array3<f64>,    // the SOM itself
    activation_map: Array2<usize>, // each cell represents how many times the corresponding cell in SOM was winner
    pub tag_map: Array2<Classification>, // each cell contains the associated classification predicted by the SOM
    tag_activation_map: Array3<usize>, // each cell represents the how many times the corresponding tag was winner for a cell
    tag_activation_map_intermed: Array2<usize>, // Identical to tag activation map but preserving a copy between unsupervised and supervised learning
    classes: HashMap<String, f64>,              // X classes with Y associated weights
    custom_weighting: bool, // Flag to enable custom weighting of classes or automatically weight by class distribution in training set
    random_seed: Option<[u8; 32]>,
}

/// A function for determining neighbours' weights.
pub type NeighbourhoodFn = fn((usize, usize), (usize, usize), f32) -> Array2<f64>;

/// A function for decaying `learning_rate` and `sigma`.
pub type DecayFn = fn(f32, u32, u32) -> f64;

pub struct SOM {
    data: SomData,
    decay_fn: DecayFn,
    neighbourhood_fn: NeighbourhoodFn,
}

// Method definitions of the SOM struct
impl SOM {
    // To create a Self-Organizing Map (SOM)
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        length: usize,
        breadth: usize,
        inputs: usize,
        randomize: bool,
        learning_rate: Option<f32>,
        sigma: Option<f32>,
        decay_fn: Option<DecayFn>,
        neighbourhood_fn: Option<NeighbourhoodFn>,
        classes: Option<HashMap<String, f64>>,
        custom_weighting: Option<bool>,
        random_seed: Option<[u8; 32]>,
    ) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not
        let the_map = if randomize {
            match random_seed {
                Some(s) => {
                    let mut rng: StdRng = SeedableRng::from_seed(s);
                    let randfn =|| {
                        rng.gen_range(0.0, 1.0) as f64
                    };
                    Array3::from_shape_simple_fn((length, breadth, inputs), randfn)
                }
                None => {
                    Array3::from_shape_simple_fn((length, breadth, inputs), random)
                }
            }
        } else {
            Array3::zeros((length, breadth, inputs))
        };
        let act_map = Array2::zeros((length, breadth));
        let tag_map = Array2::from_elem((length, breadth), Classification::default());
        let data = SomData {
            x: length,
            y: breadth,
            z: inputs,
            learning_rate: learning_rate.unwrap_or(0.5),
            maximum_iterations: 0,
            sigma: sigma.unwrap_or(0.5),
            activation_map: act_map,
            map: the_map,
            tag_map,
            tag_activation_map: {
                if let Some(c) = &classes {
                    Array3::zeros((length, breadth, c.keys().count()))
                } else {
                    Array3::zeros((length, breadth, 0))
                }
            },
            tag_activation_map_intermed: Array2::zeros((length, breadth)),
            classes: classes.unwrap_or(HashMap::new()),
            regulate_lrate: 0,
            custom_weighting: {
                if let Some(w) = custom_weighting {
                    w
                } else {
                    false
                }
            },
            random_seed,
        };
        SOM {
            data,
            decay_fn: decay_fn.unwrap_or(default_decay_fn),
            neighbourhood_fn: neighbourhood_fn.unwrap_or(gaussian),
        }
    }

    // To find and return the position of the winner neuron for a given input sample.
    //
    // TODO: (breaking-change) switch `elem` to `ArrayView1`. See todo
    //       for `Self::winner_dist()`.
    pub fn winner(&self, sample: Array1<f64>) -> (usize, usize) {
        let mut temp: Array1<f64> = Array1::<f64>::zeros(self.data.z);
        let mut min: f64 = std::f64::MAX;
        let mut ret: (usize, usize) = (0, 0);

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                for k in 0..self.data.z {
                    temp[k] = self.data.map[[i, j, k]] - sample[[k]];
                }

                let distance = distance(temp.view());

                if distance < min {
                    min = distance;
                    ret = (i, j);
                }
            }
        }
        ret
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.data)
    }

    // Update the weights of the SOM
    fn update(&mut self, elem: Array1<f64>, winner: (usize, usize), iteration_index: u32) {
        let new_lr = (self.decay_fn)(
            self.data.learning_rate,
            iteration_index,
            self.data.maximum_iterations,
        );
        let new_sig = (self.decay_fn)(
            self.data.sigma,
            iteration_index,
            self.data.maximum_iterations,
        );
        let g =
            (self.neighbourhood_fn)((self.data.x, self.data.y), winner, new_sig as f32) * new_lr;

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] += (elem[[k]] - self.data.map[[i, j, k]]) * g[[i, j]];
                }

                let distance =
                    distance(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j));
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] /= distance;
                }
            }
        }
    }

    // Update the weights of the SOM
    fn update_supervised(&mut self, elem: Classification, winner: (usize, usize), iteration_index: u32) {
        let new_lr = (self.decay_fn)(
            self.data.learning_rate,
            iteration_index,
            self.data.maximum_iterations,
        );
        let new_sig = (self.decay_fn)(
            self.data.sigma,
            iteration_index,
            self.data.maximum_iterations,
        );

        let g =
            (self.neighbourhood_fn)((self.data.x, self.data.y), winner, new_sig as f32) * new_lr;
        let winner_weight = self.data.tag_map[[winner.0, winner.1]].global_weight;
        let rand_matrix: Array2<f64> = match self.data.random_seed {
            Some(s) => {
                let mut rng: StdRng = SeedableRng::from_seed(s);
                let randfn =|| {
                    rng.gen_range(0.0, 1.0) as f64
                };
                Array2::from_shape_simple_fn((self.data.x, self.data.y), randfn)
            }
            None => {
                Array2::from_shape_simple_fn((self.data.x, self.data.y), random)
            }
        };

        let class_change_matrix = g * winner_weight;

        let modified_classes = class_change_matrix / rand_matrix;

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                if modified_classes[[i, j]] > 1.0 {
                    self.data.tag_map[[i, j]] = elem.clone();
                }
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random(&mut self, data: Array2<f64>, iterations: u32) {
        self.data.maximum_iterations = iterations;
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        let mut temp3: Array1<f64>;
        self.update_regulate_lrate(iterations);
        let mut rng: StdRng = match self.data.random_seed {
            Some(s) => {
                let r: StdRng = SeedableRng::from_seed(s);
                r
            }
            None => {
                let r: StdRng = SeedableRng::from_rng(rand::thread_rng()).unwrap();
                r
            }
        };
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp3 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            random_value = rng.gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
                temp2[i] = data[[random_value as usize, i]];
                temp3[i] = data[[random_value as usize, i]];
            }
            let win = self.winner(temp1);
            if let Some(elem) = self.data.activation_map.get_mut(win) {
                *(elem) += 1;
            }
            self.update(temp3, win, iteration);
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random_supervised(
        &mut self,
        data: Array2<f64>,
        class_data: Array1<Classification>,
        iterations: u32,
    ) {
        self.train_random(data.clone(), iterations);
        self.data.maximum_iterations = iterations;
        self.initialize_classes(data.clone(), class_data.clone());
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut ctemp1: Classification;
        self.update_regulate_lrate(iterations);
        if !self.data.custom_weighting {
            self.cal_class_weights(class_data.clone());
        }
        let mut rng: StdRng = match self.data.random_seed {
            Some(s) => {
                let r: StdRng = SeedableRng::from_seed(s);
                r
            }
            None => {
                let r: StdRng = SeedableRng::from_rng(rand::thread_rng()).unwrap();
                r
            }
        };
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            random_value = rng.gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            //random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
            }
            ctemp1 = class_data[random_value as usize].clone();
            let win = self.winner(temp1);
            if let Some(elem) = self.data.activation_map.get_mut(win) {
                *(elem) += 1;
            }
            self.update_supervised(ctemp1, win, iteration);
        }
    }

    // Initialize classes of unsupervised learner by most wins decision rule
    pub fn initialize_classes(&mut self, data: Array2<f64>, class_data: Array1<Classification>) {
        self.data.activation_map = Array2::zeros((self.data.x, self.data.y));
        let mut temp_map: HashMap<String, usize> = HashMap::new();
        let mut n = 0;
        for (k, _v) in self.data.classes.iter() {
            temp_map.insert(k.clone(), n);
            n += 1;
        }
        n = 0;
        for x in data.genrows() {
            let y = x.to_owned();
            let win = self.winner(y);
            if let Some(elem) = self.data.activation_map.get_mut(win) {
                *(elem) += 1;
            }
            if let Some(res) = temp_map.get(&class_data[n].label) {
                self.data.tag_activation_map[[win.0, win.1, *res]] += 1;
            }
            n += 1;
        }
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                let mut temp: f64 = 0.0;
                let mut class: usize = 9999;
                for k in 0..self.data.classes.keys().count() {
                    let percentage = self.data.tag_activation_map[[i, j, k]] as f64
                        / self.data.activation_map[[i, j]] as f64;
                    if percentage > temp {
                        temp = percentage;
                        class = k;
                    }
                }
                for (k, v) in temp_map.iter() {
                    if v == &class {
                        self.data.tag_map[[i, j]].label = k.clone();
                        self.data.tag_map[[i, j]].global_weight = self.data.classes.get(k).unwrap_or(&0.0).clone();
                    }
                }
            }
        }
        self.data.tag_activation_map_intermed = self.data.activation_map.clone();
    }

    // Trains the SOM given n input parameters post-training
    pub fn evaluate_hybrid(
        &mut self,
        data: Array2<f64>,
        class_data: Array1<Classification>,
    ) {
        let iterations = data.len_of(Axis(0));
        let mut temp1: Array1<f64>;
        let mut ctemp1: Classification;
        let mut temp2: Array1<f64>;
        if !self.data.custom_weighting {
            self.cal_class_weights(class_data.clone());
        }
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[iteration, i]];
                temp2[i] = data[[iteration, i]];
            }
            ctemp1 = class_data[iteration].clone();
            if ctemp1.label != "undefined".to_string() {
                let win = self.winner(temp1);
                self.update_supervised(ctemp1, win, self.data.maximum_iterations);
            } else {
                let win = self.winner(temp1);
                self.update(temp2, win, self.data.maximum_iterations);
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random_hybrid(
        &mut self,
        data: Array2<f64>,
        class_data: Array1<Classification>,
        iterations: u32,
    ) {
        self.data.maximum_iterations = iterations;
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut ctemp1: Classification;
        let mut temp2: Array1<f64>;
        if !self.data.custom_weighting {
            self.cal_class_weights(class_data.clone());
        }
        self.update_regulate_lrate(iterations);
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
                temp2[i] = data[[random_value as usize, i]];
            }
            ctemp1 = class_data[random_value as usize].clone();
            if ctemp1.label != "undefined".to_string() {
                let win = self.winner(temp1);
                self.update_supervised(ctemp1, win, iteration);
            } else {
                let win = self.winner(temp1);
                self.update(temp2, win, iteration);
            }
        }
    }
    // Trains the SOM by picking  data points in batches (sequentially) as inputs from the dataset
    pub fn train_batch(&mut self, data: Array2<f64>, iterations: u32) {
        let mut index: u32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        self.update_regulate_lrate(ndarray::ArrayBase::dim(&data).0 as u32 * iterations);
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            index = iteration % (ndarray::ArrayBase::dim(&data).0 - 1) as u32;
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[index as usize, i]];
                temp2[i] = data[[index as usize, i]];
            }
            let win = self.winner(temp1);
            self.update(temp2, win, iteration);
        }
    }

    fn cal_class_weights(&mut self, class_data: Array1<Classification>) {
        let num_classes = self.data.classes.keys().count();
        let len_data = class_data.len();
        for (c, w) in self.data.classes.iter_mut() {
            let mut temp = 0.0;
            for i in 0..class_data.dim() {
                if class_data[i].label == c.as_str() {
                    temp += 1.0;
                }
            }
            if temp > 0.0 {
                let weight = (len_data as f64) / (num_classes as f64 * temp);
                *w = weight;
            } else {
                *w = 0.0;
            }
        }
    }

    // Update learning rate regulator (keep learning rate constant with increase in number of iterations)
    fn update_regulate_lrate(&mut self, iterations: u32) {
        self.data.regulate_lrate = iterations / 2;
    }

    // Returns the activation map of the SOM, where each cell at (i, j) represents how many times the cell at (i, j) in the SOM was picked a winner neuron.
    pub fn activation_response(&self) -> ArrayView2<usize> {
        self.data.activation_map.view()
    }

    // Similar to winner(), but also returns distance of input sample from winner neuron.
    //
    // TODO: (breaking-change) make `elem` an `ArrayView1` to remove
    //       at least one heap allocation. Requires same change to
    //       `Self::winner()`.
    //
    pub fn winner_dist(&self, elem: Array1<f64>) -> ((usize, usize), f64) {
        // TODO: use more descriptive names than temp[..]
        let tempelem = elem.clone();
        let temp = self.winner(elem);

        (
            temp,
            euclid_dist(
                self.data
                    .map
                    .index_axis(Axis(0), temp.0)
                    .index_axis(Axis(0), temp.1),
                tempelem.view(),
            ),
        )
    }

    /// Returns values associated with winner node from map and tag_map
    pub fn winner_vals(&self, elem: Array1<f64>) -> (((usize, usize), f64), String) {
        // TODO: use more descriptive names than temp[..]
        let temp = self.winner(elem.clone());
        self.data.tag_map.index_axis(Axis(0), temp.0).index_axis(Axis(0), temp.1);
        // TODO: Get rid of this clone!
        (self.winner_dist(elem), self.data.tag_map[[temp.0, temp.1]].label.clone())
    }
    // Returns size of SOM.
    pub fn get_size(&self) -> (usize, usize) {
        (self.data.x, self.data.y)
    }

    // Returns the distance map of each neuron / the normalised sum of a neuron to every other neuron in the map.
    pub fn distance_map(&self) -> Array2<f64> {
        let mut dist_map = Array2::<f64>::zeros((self.data.x, self.data.y));
        let mut temp_dist: f64;
        let mut max_dist: f64 = 0.0;
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                temp_dist = 0.0;
                for k in 0..self.data.x {
                    for l in 0..self.data.y {
                        temp_dist += euclid_dist(
                            self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j),
                            self.data.map.index_axis(Axis(0), k).index_axis(Axis(0), l),
                        );
                    }
                }
                if temp_dist > max_dist {
                    max_dist = temp_dist;
                }
                dist_map[[i, j]] = temp_dist;
            }
        }
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                dist_map[[i, j]] /= max_dist;
            }
        }
        dist_map
    }

    // Unit testing functions for setting individual cell weights
    #[cfg(test)]
    pub fn set_map_cell(&mut self, (i, j, k): (usize, usize, usize), val: f64) {
        self.data.map[[i, j, k]] = val;
    }

    // Unit testing functions for getting individual cell weights
    #[cfg(test)]
    pub fn get_map_cell(&self, (i, j, k): (usize, usize, usize)) -> f64 {
        self.data.map[[i, j, k]]
    }
}

// To enable SOM objects to be printed with "print" and it's family of formatted string printing functions
impl fmt::Display for SOM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mut i, mut j) = (0, 0);

        for vector in self.data.map.lanes(Axis(2)) {
            println!("[{}, {}] : {}", i, j, vector);

            j += 1;
            if j == self.data.y {
                j = 0;
                i += 1;
            }
        }

        write!(f, "\nSOM Shape = ({}, {})\nExpected input vectors of length = {}\nSOM learning rate regulator = {}", self.data.x, self.data.y, self.data.z, self.data.regulate_lrate)
    }
}

// Returns the 2-norm of a vector represented as a 1D ArrayView
pub fn distance(a: ArrayView1<f64>) -> f64 {
    a.iter().map(|elem| elem.powi(2)).sum::<f64>().sqrt()
}

// The default decay function for LR and Sigma
pub fn default_decay_fn(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    val as f64 * (-(curr_iter as f64 / max_iter as f64)).exp()
}

#[allow(unused)]
pub fn exponential_decay_fn(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    (val as f64).powf((curr_iter + 1) as f64 / (max_iter + 1) as f64)
}

/// Default neighborhood function.
///
/// Returns a two-dimensional Gaussian distribution centered at `pos`.
pub fn gaussian(dims: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {
    let div = 2.0 * PI * (sigma as f64).powi(2);

    let shape_fn = |(i, j)| {
        let x = (-(((i as f64 - (pos.0 as f64)).powi(2)).sqrt() / div)).exp();
        let y = (-(((j as f64 - (pos.1 as f64)).powi(2)).sqrt() / div)).exp();
        x * y
    };

    Array2::from_shape_fn(dims, shape_fn)
}

/// Mexican-Hat neighborhood function
///
/// Returns a two-dimensional Gaussian distribution centered at `pos`.
#[allow(unused)]
pub fn mh_neighborhood(dims: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {
    let div = 2.0 * PI * (sigma as f64).powi(2);
    let shape_fn = |(i, j)| {
        let d = (((i as f64 - pos.0 as f64) + (j as f64 - pos.1 as f64)).powi(2)).sqrt();
        let x = (-(((i as f64 - (pos.0 as f64)).powi(2)).sqrt() / div)).exp();
        let y = (-(((j as f64 - (pos.1 as f64)).powi(2)).sqrt() / div)).exp();
        (1.0 - (d.powi(2) as f64) / (sigma.powi(2) as f64)) * (x * y)
    };

    Array2::from_shape_fn(dims, shape_fn)
}

/// Returns the [Euclidean distance] between `a` and `b`.
///
/// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
fn euclid_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn from_json(
    serialized: &str,
    decay_fn: Option<DecayFn>,
    neighbourhood_fn: Option<NeighbourhoodFn>,
) -> serde_json::Result<SOM> {
    let data: SomData = serde_json::from_str(&serialized)?;
    Ok(SOM {
        data,
        decay_fn: decay_fn.unwrap_or(default_decay_fn),
        neighbourhood_fn: neighbourhood_fn.unwrap_or(gaussian),
    })
}

// Unit-testing module - only compiled when "cargo test" is run!
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_winner() {
        let mut map = SOM::create(
            2,
            3,
            5,
            false,
            Some(0.1),
            None,
            None,
            None,
            None,
            None,
            Some([1,2,3,4,5,6,7,8,9,10,11,1,2,3,4,5,6,7,8,9,10,11,1,2,3,4,5,6,7,8,9,10]));

        for k in 0..5 {
            map.set_map_cell((1, 1, k), 1.5);
        }

        for k in 0..5 {
            assert_eq!(map.get_map_cell((1, 1, k)), 1.5);
        }

        assert_eq!(map.winner(Array1::from(vec![1.5; 5])), (1, 1));
        assert_eq!(map.winner(Array1::from(vec![0.5; 5])), (0, 0));
    }

    #[test]
    fn test_euclid() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![4.0, 5.0, 6.0, 7.0];

        assert_eq!(euclid_dist(a.view(), b.view()), 6.0);
    }
}
