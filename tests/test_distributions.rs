use csv;
use ndarray::{Array1, Array2};
use rusticsom::SOM;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use stopwatch::Stopwatch;

#[derive(Serialize, Deserialize, Debug, Default)]
struct TestSamples {
    #[serde(alias = "")]
    pub pos: String,
    pub rw_distance: f64,
    pub signal_mean: f64,
    pub snr_mean: f64,
    pub fspl: f64,
    pub first_sig_sd_per: f64,
    pub left_sig_sd_per: f64,
    pub right_sig_sd_per: f64,
    pub first_snr_sd_per: f64,
    pub left_snr_sd_per: f64,
    pub right_snr_sd_per: f64,
    pub label: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct TestSamplesTag {
    pub pos: String,
    pub rw_distance: f64,
    pub signal: f64,
    pub snr: f64,
    pub fspl: f64,
    pub first_sig_per: f64,
    pub left_sig_per: f64,
    pub right_sig_per: f64,
    pub first_snr_per: f64,
    pub left_snr_per: f64,
    pub right_snr_per: f64,
    pub label: String,
}

#[test]
fn test_dist_unsupervised() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples_6.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            rw_distance: record.rw_distance,
            signal: record.signal_mean,
            snr: record.snr_mean,
            first_sig_per: record.first_sig_sd_per,
            left_sig_per: record.left_sig_sd_per,
            right_sig_per: record.right_sig_sd_per,
            first_snr_per: record.first_snr_sd_per,
            left_snr_per: record.left_snr_sd_per,
            right_snr_per: record.right_snr_sd_per,
            fspl: record.fspl,
            label: record.label,
        };
        samples.push(rec_tag);
    }

    // Format parsed training samples
    let testsamplen = samples.len();
    let mut fmtdata: Vec<f64> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();
    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(data.signal);
        fmtdata.push(data.snr);
        fmtdata.push(data.first_snr_per);
        fmtdata.push(data.left_snr_per);
        fmtdata.push(data.right_snr_per);
        fmtdata.push(data.fspl);
        fmtdata.push(data.rw_distance);
        fmtclass.push(data.label.clone());
    }

    // Initialize classes if there are any
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("positive".to_string(), 0.0);
    classes.insert("negative".to_string(), 0.0);
    classes.insert("middleman".to_string(), 0.0);
    let classes = classes.clone();

    // Create a new SOM using default settings
    let mut map = SOM::create(
        20,
        20,
        7,
        true,
        Some(0.5),
        Some(0.5),
        None,
        None,
        Some(classes),
        None,
        Some([209,162,182,84,44,167,62,240,152,122,118,154,48,208,143,84,
            186,211,219,113,71,108,171,185,51,159,124,176,167,192,23,245])
    );
    let newdat = Array2::from_shape_vec((fmtdata.len()/7, 7), fmtdata).unwrap();
    let newdat2 = newdat.clone();
    // Unsupervised training of the SOM
    let sw = Stopwatch::start_new();
    map.train_random(newdat, 20000);
    println!("TrainUnsupervised: {:?}", sw.elapsed());

    // Output relative distance map of SOM nodes
    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    // Write trained SOM results to file for plotting externally
    let mut file = File::create("outputs/output_large_unsupervised.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    // Write out condensed winner list
    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}

#[test]
fn test_distributions() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples_6.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            rw_distance: record.rw_distance,
            signal: record.signal_mean,
            snr: record.snr_mean,
            first_sig_per: record.first_sig_sd_per,
            left_sig_per: record.left_sig_sd_per,
            right_sig_per: record.right_sig_sd_per,
            first_snr_per: record.first_snr_sd_per,
            left_snr_per: record.left_snr_sd_per,
            right_snr_per: record.right_snr_sd_per,
            fspl: record.fspl,
            label: record.label,
        };
        samples.push(rec_tag);
    }

    let testsamplen = samples.len();
    let mut fmtdata: Vec<f64> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();

    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(data.signal);
        fmtdata.push(data.snr);
        fmtdata.push(data.first_snr_per);
        fmtdata.push(data.left_snr_per);
        fmtdata.push(data.right_snr_per);
        fmtdata.push(data.fspl);
        fmtdata.push(data.rw_distance);
        fmtclass.push(data.label.clone());
    }

    // Initialize classes if there are any
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("positive".to_string(), 0.0);
    classes.insert("negative".to_string(), 0.0);
    classes.insert("middleman".to_string(), 0.0);
    let classes = classes.clone();

    let mut map = SOM::create(
        20,
        20,
        7,
        true,
        Some(0.5),
        Some(0.5),
        None,
        None,
        Some(classes),
        None,
        Some([209,162,182,84,44,167,62,240,152,122,118,154,48,208,143,84,
            186,211,219,113,71,108,171,185,51,159,124,176,167,192,23,245])
    );
    let newdat = Array2::from_shape_vec((fmtdata.len()/7, 7), fmtdata).unwrap();
    let newdat2 = newdat.clone();
    let newlabel = Array1::from(fmtclass);

    // Run through supervised training method
    let sw = Stopwatch::start_new();
    map.train_random_supervised(newdat, newlabel, 20000);
    println!("TrainSupervised: {:?}", sw.elapsed());

    let mut file = File::create("outputs/output_large_supervised.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}
