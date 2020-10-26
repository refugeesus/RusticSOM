/// Core SOM definitions and functionality
///
pub mod functions;
use rand;
use ndarray;
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use ndarray::{Array1, Array2, Array3, Axis, ArrayView1, ArrayView2};
use std::fmt;
use crate::functions::distance::euclid_dist;
use crate::functions::neighbourhood::gaussian;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DecayFunction {
    Exponential,
    Power,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActivationMap {
    act_map: Array2<usize>,
    tag_wins: Array3<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SomData {
    x: usize,               // length of SOM
    y: usize,               // breadth of SOM
    z: usize,               // size of inputs
    sigma: f64,           // spread of neighbourhood function, default = 1.0
    //regulate_lrate: u32,    // Regulates the learning rate w.r.t the number of iterations
    learning_rate: (f64, f64),
    current_lr: f64,
    classes: Vec<String>,
    class_map: HashMap<String, f64>,
    pub map: Array3<f64>,       // the SOM itself
    pub tag_map: Array1<String>,
    pub activation_map: ActivationMap,
}

pub struct SOM {
    pub data: SomData,
    //decay_function: fn(f64, u32, u32) -> f64,          // the function used to decay learning_rate and sigma
    decay_function: DecayFunction,
    neighbourhood_function: fn((usize, usize), (usize, usize), f32) -> Array2<f64>,          // the function that determines the weights of the neighbours
}

// Method definitions of the SOM struct
impl SOM {
    // To create a Self-Organizing Map (SOM)
    pub fn create(
        length: usize,
        breadth: usize,
        inputs: usize,
        classes: Vec<String>,
        rand_range: Option<(f64,f64)>,
        learning_rate: Option<f64>,
        sigma: Option<f64>,
        //decay_function: Option<fn(f64, u32, u32) -> f64>,
        decay_function: Option<DecayFunction>,
        neighbourhood_function: Option<fn((usize, usize), (usize, usize), f32) -> Array2<f64>>) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not
        let mut the_map = Array3::<f64>::zeros((length, breadth, inputs));

        let act_map_vals = Array2::<usize>::zeros((length, breadth));
        let act_map_wins = Array3::<usize>::zeros((length, breadth, &classes.len()+1));

        let mut temp_classes: Vec<String> = classes.clone();
        temp_classes.push("none".to_string());

        let act_map = ActivationMap {
            act_map: act_map_vals,
            tag_wins: act_map_wins,
        };

        let tag_map = Array1::<String>::from_elem(length*breadth, "none".to_string());
        let mut _init_regulate_lrate = 0;

        match rand_range {
            Some(b) => {
                let mut rng = rand::thread_rng();
                let uniform = Uniform::new(b.0, b.1);
                for element in the_map.iter_mut() {
                    *element = uniform.sample(&mut rng);
                }
                //for element in the_map.iter_mut() {
                //    *element = random::<f64>();
                //}
            }
            None => {
                // Do not randomize any of the map initial weights... for whatever reason someone
                // would do this I guess
                ()
            }
        }
        let mut init_weights = HashMap::new();
        for class in classes {
            init_weights.insert(class.clone(), 1.0);
        }
        init_weights.insert("none".to_string(), 1.0);
        let data = SomData {
            x: length,
            y: breadth,
            z: inputs,
            learning_rate: match learning_rate {
                None => (0.5, 0.01),
                Some(value) => (value as f64, 0.01),
            },
            current_lr: match learning_rate {
                None => 0.5,
                Some(value) => value,
            },
            sigma: match sigma {
                None => length as f64*(2/3) as f64,
                Some(value) => value,
            },
            activation_map: act_map,
            map: the_map,
            tag_map,
            classes: temp_classes,
            class_map: init_weights,
            //regulate_lrate: _init_regulate_lrate,
        };
        SOM {
            data,
            decay_function: match decay_function {
                None => DecayFunction::Exponential,
                Some(f) => f,
            },
            neighbourhood_function: match neighbourhood_function {
                None => gaussian,
                Some(foo) => foo,
            },
        }
    }

    // To find and return the position of the winner neuron for a given input sample.
    pub fn winner(&mut self, elem: Array1<f64>, sample_class: Option<String>) -> ((usize, usize), Option<String>) {
        let mut temp: Array1<f64> = Array1::<f64>::zeros(self.data.z);
        let mut min: f64 = std::f64::MAX;
        let mut ret: (usize, usize) = (0, 0);
        let mut wclass: Option<String> = None;
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                for k in 0..self.data.z {
                    temp[k] = self.data.map[[i, j, k]] - elem[[k]];
                }

                // This is effectively the same as the euclid_dist fn... why are they separate?
                let norm = norm(temp.view());

                if norm < min {
                    min = norm;
                    ret = (i, j);
                    wclass = Some(self.data.tag_map[i + self.data.x*j].clone());
                }
            }
        }

        if let Some(elem) = self.data.activation_map.act_map.get_mut(ret) {
            *(elem) += 1;
        }

        let tmp_class = sample_class.unwrap_or("none".to_string());
        for c in 0..self.data.classes.len() {
            if self.data.classes[c] == tmp_class {
                self.data.activation_map.tag_wins[[ret.0,ret.1,c]] += 1;
            }
        }

        (ret, wclass)
    }

    pub fn initialize_classes(&mut self) {
        let sum = self.data.activation_map.tag_wins.sum_axis(Axis(2));
        //println!("{:?}", sum.view());
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                let mut tmp_max: f64 = 0.0;
                for k in 0..self.data.classes.len() {
                    let max = self.data.activation_map.tag_wins[[i, j, k]] as f64 / sum[[i,j]] as f64;
                    if max > tmp_max {
                        tmp_max = max;
                        self.data.tag_map[i + self.data.x*j] = self.data.classes.get(k).unwrap().clone();
                    }
                    // TODO: Need to cover case tmp == max
                }
            }
        }
    }

    pub fn update_supervised(&mut self, winner: (usize, usize), sample_class: String, iteration_index: u32, total_iterations: u32) {
        let (new_lr, new_sig) = match self.decay_function {
            DecayFunction::Exponential => {
                (functions::exponential_decay(self.data.current_lr, iteration_index, total_iterations),
                functions::exponential_decay(self.data.sigma, iteration_index, total_iterations))
            }
            DecayFunction::Power => {
                (functions::power_decay(self.data.learning_rate.0, self.data.learning_rate.1, iteration_index, total_iterations),
                functions::power_decay(self.data.sigma, 0.0, iteration_index, total_iterations))
            }
        };
        //let new_lr = (self.decay_function)(self.data.learning_rate.0, iteration_index, total_iterations);
        //let new_sig = (self.decay_function)(self.data.sigma, iteration_index, total_iterations);

        let g = (self.neighbourhood_function)((self.data.x, self.data.y), winner, new_sig as f32) * new_lr;

        let winner_weight = self.data.class_map.get(&self.data.tag_map[winner.0 + self.data.x*winner.1]).unwrap_or(&1.0).to_owned();
        let mut rand_matrix: Array1<f64> = Array1::<f64>::zeros(self.data.x*self.data.y);
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);
        for element in rand_matrix.iter_mut() {
            *element = uniform.sample(&mut rng);
        }
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                let prob_change = g[[i, j]] * winner_weight;
                if prob_change > rand_matrix[i + self.data.x*j] {
                    self.data.tag_map[i + self.data.x*j] = sample_class.clone();
                }

                let norm = norm(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j));
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] /= norm;
                }
            }
        }
        self.data.current_lr = new_lr;
    }

    // Update the weights of the SOM
    fn update_unsupervised(&mut self, elem: Array1<f64>, winner: (usize, usize), iteration_index: u32, total_iterations: u32) {
        let (new_lr, new_sig) = match self.decay_function {
            DecayFunction::Exponential => {
                (functions::exponential_decay(self.data.current_lr, iteration_index, total_iterations),
                functions::exponential_decay(self.data.sigma, iteration_index, total_iterations))
            }
            DecayFunction::Power => {
                (functions::power_decay(self.data.learning_rate.0, self.data.learning_rate.1, iteration_index, total_iterations),
                functions::power_decay(self.data.sigma, 0.0, iteration_index, total_iterations))
            }
        };
        let g = (self.neighbourhood_function)((self.data.x, self.data.y), winner, new_sig as f32) * new_lr;

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] += (elem[[k]] - self.data.map[[i, j, k]]) * g[[i, j]];
                }

                let norm = norm(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j));
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] /= norm;
                }
            }
        }
        self.data.current_lr = new_lr;
    }

    pub fn from_json(
        serialized: &str,
        decay_function: Option<DecayFunction>,
        //decay_function: Option<fn(f64, u32, u32) -> f64>,
        neighbourhood_function: Option<fn((usize, usize), (usize, usize), f32) -> Array2<f64>>) -> serde_json::Result<SOM> {
        let data: SomData = serde_json::from_str(&serialized)?;

        Ok(SOM {
            data,
            decay_function: match decay_function {
                None => DecayFunction::Exponential,
                Some(f) => f,
            },
            neighbourhood_function: match neighbourhood_function {
                None => gaussian,
                Some(foo) => foo,
            },
        })
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.data)
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random(&mut self, data: Array2<f64>, iterations: u32) {
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        //self.update_regulate_lrate(iterations);
        for iteration in 0..iterations{
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
                temp2[i] = data[[random_value as usize, i]];
            }
            let (win, _class) = self.winner(temp1, None);
            self.update_unsupervised(temp2, win, iteration, iterations);
        }
    }

    pub fn train_random_hybrid(&mut self, data: Array2<f64>, class_data: Option<Array1<String>>, iterations: u32) {
        // Extract class data if there is some
        if let Some(cdata) = class_data {
            let mut random_value: i32;
            let mut temp1: Array1<f64>;
            let mut temp2: Array1<f64>;

            let mut ctemp1: String;
            let mut ctemp2: String;

            self.cal_class_weights(&cdata);

            for iteration in 0..iterations {
                // Temporary values for selected sample in training data.
                temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
                temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
                random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);

                // Get random entry from training data
                for i in 0..ndarray::ArrayBase::dim(&data).1 {
                    temp1[i] = data[[random_value as usize, i]];
                    temp2[i] = data[[random_value as usize, i]];
                }
                ctemp1 = cdata[random_value as usize].to_owned();
                ctemp2 = cdata[random_value as usize].to_owned();

                // Check if the random entry from training data has a classification or not
                if ctemp1 != "none".to_string() {
                    // We have a classification, update as supervised
                    let (win, _) = self.winner(temp1, Some(ctemp1));
                    self.update_supervised(win, ctemp2, iteration, iterations);
                } else {
                    // No classification here, update as unsupervised
                    let (win, _class) = self.winner(temp1, None);
                    self.update_unsupervised(temp2, win, iteration, iterations);
                }
            }
        } else {
            // We don't have any class data for this training session. Run as unsupervised for the
            // whole thing.
            let mut random_value: i32;
            let mut temp1: Array1<f64>;
            let mut temp2: Array1<f64>;
            //self.update_regulate_lrate(iterations);
            for iteration in 0..iterations{
                temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
                temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
                random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
                for i in 0..ndarray::ArrayBase::dim(&data).1 {
                    temp1[i] = data[[random_value as usize, i]];
                    temp2[i] = data[[random_value as usize, i]];
                }
                let (win, _class) = self.winner(temp1, None);
                self.update_unsupervised(temp2, win, iteration, iterations);
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random_supervised(&mut self, data: Array2<f64>, class_data: Array1<String>, iterations: u32) {
        //self.initialize_classes();
        //println!("{:?}", self.data.tag_map.view());
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut class_temp: String;
        let mut class_temp2: String;
        let mut temp2: Array1<f64>;
        //self.update_regulate_lrate(iterations);
        self.cal_class_weights(&class_data);
        for iteration in 0..iterations{
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            //class_temp1 = Array1::<String>::from_elem(ndarray::ArrayBase::dim(&class_data), "none".to_string());
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
                temp2[i] = data[[random_value as usize, i]];
            }
            class_temp = class_data[random_value as usize].to_owned();
            class_temp2 = class_data[random_value as usize].to_owned();
            let (win, _win_class) = self.winner(temp1, Some(class_temp));
            //self.update(temp2, win, iteration, iterations);
            self.update_supervised(win, class_temp2, iteration, iterations);
        }
        self.initialize_classes();
    }

    // Trains the SOM by picking  data points in batches (sequentially) as inputs from the dataset
    pub fn train_batch(&mut self, data: Array2<f64>, iterations: u32) {
        let mut index: u32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        //self.update_regulate_lrate(ndarray::ArrayBase::dim(&data).0 as u32 * iterations);
        for iteration in 0..iterations{
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            index = iteration % (ndarray::ArrayBase::dim(&data).0 - 1) as u32;
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[index as usize, i]];
                temp2[i] = data[[index as usize, i]];
            }
            let (win, _win_class) = self.winner(temp1, None);
            self.update_unsupervised(temp2, win, iteration, iterations);
        }
    }

    fn cal_class_weights(&mut self, class_data: &Array1<String>) {
        let num_classes = self.data.class_map.keys().count();
        let len_data = class_data.len();
        for (c, w) in self.data.class_map.iter_mut() {
            let mut temp = 0.0;
            for i in 0..self.data.x*self.data.y {
                if class_data[i] == c.as_str() {
                    temp += 1.0;
                }
            }
            let weight = (len_data as f64)/(num_classes as f64 * temp);
            *w = weight;
        }
    }

    // Update learning rate regulator (keep learning rate constant with increase in number of iterations)
    //fn update_regulate_lrate(&mut self, iterations: u32){
    //    self.data.regulate_lrate = iterations / 2;
    //}

    // Returns the activation map of the SOM, where each cell at (i, j) represents how many times the cell at (i, j) in the SOM was picked a winner neuron.
    pub fn activation_response(&self) -> ArrayView2<usize> {
        self.data.activation_map.act_map.view()
    }

    // Similar to winner(), but also returns distance of input sample from winner neuron.
    pub fn winner_dist(&mut self, elem: Array1<f64>) -> ((usize, usize), f64, String) {
        let mut tempelem = Array1::<f64>::zeros(elem.len());

        for i in 0..elem.len() {
            if let Some(temp) = tempelem.get_mut(i) {
                *(temp) = elem[i];
            }
        }

        let (temp, class) = self.winner(elem, None);

        (temp, euclid_dist(self.data.map.index_axis(Axis(0), temp.0).index_axis(Axis(0), temp.1), tempelem.view()), class.unwrap_or("NA".to_string()))
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
                for k in 0..self.data.x{
                    for l in 0..self.data.y{
                        temp_dist += euclid_dist(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j), self.data.map.index_axis(Axis(0), k).index_axis(Axis(0), l));
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
        return dist_map;
    }

    // Unit testing functions for setting individual cell weights
    #[cfg(test)]
    pub fn set_map_cell(&mut self, pos: (usize, usize, usize), val: f64) {
        if let Some(elem) = self.data.map.get_mut(pos) {
             *(elem) = val;
        }
    }

    // Unit testing functions for getting individual cell weights
    #[cfg(test)]
    pub fn get_map_cell(&self, pos: (usize, usize, usize)) -> f64 {
        if let Some(elem) = self.data.map.get(pos) {
             *(elem)
        }
        else {
            panic!("Invalid index!");
        }
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

        write!(f, "\nSOM Shape = ({}, {})\nExpected input vectors of length = {}\nSOM learning rate regulator = {}", self.data.x, self.data.y, self.data.z, self.data.current_lr)
    }
}

// Returns the 2-norm of a vector represented as a 1D ArrayView
fn norm(a: ArrayView1<f64>) -> f64 {
    let mut ret: f64 = 0.0;

    for i in a.iter() {
        ret += i.powf(2.0);
    }

    ret.powf(0.5)
}

// The default decay function for LR and Sigma
/*
fn default_decay_function(val: f64, curr_iter: u32, max_iter: u32) -> f64 {
    val / ((1 + (curr_iter/max_iter)) as f64)
}
*/
// Unit-testing module - only compiled when "cargo test" is run!
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winner() {
        let mut classes: Vec<String> = Vec::new();
        classes.push("none".to_string());
        let mut map = SOM::create(2, 3, 5, classes, None, Some(0.1), None, None, None);

        for k in 0..5 {
            map.set_map_cell((1, 1, k), 1.5);
        }

        for k in 0..5 {
            assert_eq!(map.get_map_cell((1, 1, k)), 1.5);
        }

        assert_eq!(map.winner(Array1::from(vec![1.5; 5]), None), ((1, 1), None));
        // Floor not 0 with uniform distribution initialized map
        assert_eq!(map.winner(Array1::from(vec![0.5; 5]), None), ((0, 0), None));
    }

    #[test]
    fn test_euclid() {
        let a = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from(vec![4.0, 5.0, 6.0, 7.0]);

        assert_eq!(euclid_dist(a.view(), b.view()), 6.0);
    }
}
