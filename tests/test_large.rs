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
    pub challengee: String,
    pub witness_gateway: String,
    pub rw_distance: f64,
    #[serde(alias = "signal_mean.1")]
    pub signal_1: f64,
    #[serde(alias = "snr_mean.1")]
    pub snr_1: f64,
    pub fspl: f64,
    pub fspl_mw: f64,
    #[serde(alias = "signal_var.1")]
    pub signal_var_1: f64,
    #[serde(alias = "snr_var.1")]
    pub snr_var_1: f64,
    #[serde(alias = "signal_mean.2")]
    pub signal_2: f64,
    #[serde(alias = "snr_mean.2")]
    pub snr_2: f64,
    #[serde(alias = "signal_var.2")]
    pub signal_var_2: f64,
    #[serde(alias = "snr_var.2")]
    pub snr_var_2: f64,
    #[serde(alias = "signal_mean.3")]
    pub signal_3: f64,
    #[serde(alias = "snr_mean.3")]
    pub snr_3: f64,
    #[serde(alias = "signal_var.3")]
    pub signal_var_3: f64,
    #[serde(alias = "snr_var.3")]
    pub snr_var_3: f64,
    #[serde(alias = "signal_mean.4")]
    pub signal_4: f64,
    #[serde(alias = "snr_mean.4")]
    pub snr_4: f64,
    #[serde(alias = "signal_mean_mw.1")]
    pub signal_mw_1: f64,
    #[serde(alias = "snr_mean_mw.1")]
    pub snr_mw_1: f64,
    #[serde(alias = "signal_mean_mw.2")]
    pub signal_mw_2: f64,
    #[serde(alias = "snr_mean_mw.2")]
    pub snr_mw_2: f64,
    #[serde(alias = "signal_mean_mw.3")]
    pub signal_mw_3: f64,
    #[serde(alias = "snr_mean_mw.3")]
    pub snr_mw_3: f64,
    #[serde(alias = "signal_mean_mw.4")]
    pub signal_mw_4: f64,
    #[serde(alias = "snr_mean_mw.4")]
    pub snr_mw_4: f64,
    #[serde(alias = "signal_var.4")]
    pub signal_var_4: f64,
    #[serde(alias = "snr_var.4")]
    pub snr_var_4: f64,
    pub label: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct TestSamplesTag {
    pub pos: String,
    pub challengee: String,
    pub witness_gateway: String,
    pub rw_distance: f64,
    pub signal_1: f64,
    pub snr_1: f64,
    pub fspl: f64,
    pub signal_var_1: f64,
    pub snr_var_1: f64,
    pub signal_2: f64,
    pub snr_2: f64,
    pub signal_var_2: f64,
    pub snr_var_2: f64,
    pub signal_3: f64,
    pub snr_3: f64,
    pub signal_var_3: f64,
    pub snr_var_3: f64,
    pub signal_4: f64,
    pub snr_4: f64,
    pub signal_mw_1: f64,
    pub snr_mw_1: f64,
    pub signal_mw_2: f64,
    pub snr_mw_2: f64,
    pub signal_mw_3: f64,
    pub snr_mw_3: f64,
    pub signal_mw_4: f64,
    pub snr_mw_4: f64,
    pub signal_var_4: f64,
    pub snr_var_4: f64,
    pub label: String,
}

#[test]
fn test_large_unsupervised() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/normalized_samples_trimmed.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            rw_distance: record.rw_distance,

            signal_1: record.signal_1,
            snr_1: record.snr_1,
            signal_mw_1: record.signal_mw_1,
            snr_mw_1: record.snr_mw_1,
            signal_var_1: record.signal_var_1,
            snr_var_1: record.snr_var_1,

            signal_2: record.signal_2,
            snr_2: record.snr_2,
            signal_mw_2: record.signal_mw_2,
            snr_mw_2: record.snr_mw_2,
            signal_var_2: record.signal_var_2,
            snr_var_2: record.snr_var_2,

            signal_3: record.signal_3,
            snr_3: record.snr_3,
            signal_mw_3: record.signal_mw_3,
            snr_mw_3: record.snr_mw_3,
            signal_var_3: record.signal_var_3,
            snr_var_3: record.snr_var_3,

            signal_4: record.signal_4,
            snr_4: record.snr_4,
            signal_mw_4: record.signal_mw_4,
            snr_mw_4: record.snr_mw_4,
            signal_var_4: record.signal_var_4,
            snr_var_4: record.snr_var_4,

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
        fmtdata.push(data.signal_1);
        fmtdata.push(data.snr_1);
        fmtdata.push(data.signal_var_1);
        fmtdata.push(data.snr_var_1);
        //fmtdata.push(data.signal_2);
        //fmtdata.push(data.snr_2);
        fmtdata.push(data.signal_var_2);
        fmtdata.push(data.snr_var_2);
        //fmtdata.push(data.signal_3);
        //fmtdata.push(data.snr_3);
        fmtdata.push(data.signal_var_3);
        fmtdata.push(data.snr_var_3);
        fmtdata.push(data.signal_4);
        fmtdata.push(data.snr_4);
        fmtdata.push(data.signal_var_4);
        fmtdata.push(data.snr_var_4);
        fmtdata.push(data.fspl);
        fmtdata.push(data.rw_distance);
        /*
        fmtdata.push(
            [data.signal_1, data.snr_1,
            data.signal_var_1, data.snr_var_1,
            data.signal_2, data.snr_2,
            data.signal_var_2, data.snr_var_2,
            data.signal_3, data.snr_3,
            data.signal_var_3, data.snr_var_3,
            data.signal_4, data.snr_4,
            data.signal_var_4, data.snr_var_4,
            data.fspl, data.rw_distance]);
        */
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
        14,
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
    let newdat = Array2::from_shape_vec((fmtdata.len()/14, 14), fmtdata).unwrap();
    let newdat2 = newdat.clone();
    // Unsupervised training of the SOM
    let sw = Stopwatch::start_new();
    map.train_random(newdat, 10000);
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
fn test_large_supervised() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/normalized_samples_trimmed.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            rw_distance: record.rw_distance,

            signal_1: record.signal_1,
            snr_1: record.snr_1,
            signal_mw_1: record.signal_mw_1,
            snr_mw_1: record.snr_mw_1,
            signal_var_1: record.signal_var_1,
            snr_var_1: record.snr_var_1,

            signal_2: record.signal_2,
            snr_2: record.snr_2,
            signal_mw_2: record.signal_mw_2,
            snr_mw_2: record.snr_mw_2,
            signal_var_2: record.signal_var_2,
            snr_var_2: record.snr_var_2,

            signal_3: record.signal_3,
            snr_3: record.snr_3,
            signal_mw_3: record.signal_mw_3,
            snr_mw_3: record.snr_mw_3,
            signal_var_3: record.signal_var_3,
            snr_var_3: record.snr_var_3,

            signal_4: record.signal_4,
            snr_4: record.snr_4,
            signal_mw_4: record.signal_mw_4,
            snr_mw_4: record.snr_mw_4,
            signal_var_4: record.signal_var_4,
            snr_var_4: record.snr_var_4,

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
        fmtdata.push(data.signal_1);
        fmtdata.push(data.snr_1);
        fmtdata.push(data.signal_var_1);
        fmtdata.push(data.snr_var_1);
        //fmtdata.push(data.signal_2);
        //fmtdata.push(data.snr_2);
        fmtdata.push(data.signal_var_2);
        fmtdata.push(data.snr_var_2);
        //fmtdata.push(data.signal_3);
        //fmtdata.push(data.snr_3);
        fmtdata.push(data.signal_var_3);
        fmtdata.push(data.snr_var_3);
        fmtdata.push(data.signal_4);
        fmtdata.push(data.snr_4);
        fmtdata.push(data.signal_var_4);
        fmtdata.push(data.snr_var_4);
        fmtdata.push(data.fspl);
        fmtdata.push(data.rw_distance);
/*
        fmtdata.push(
            [data.signal_1, data.snr_1,
            data.signal_var_1, data.snr_var_1,
            data.signal_2, data.snr_2,
            data.signal_var_2, data.snr_var_2,
            data.signal_3,
            data.signal_var_3, data.snr_var_3,
            data.signal_4, data.snr_4,
            data.signal_var_4, data.snr_var_4,
            data.fspl]);
*/
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
        14,
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
    let newdat = Array2::from_shape_vec((fmtdata.len()/14, 14), fmtdata).unwrap();
    let newdat2 = newdat.clone();
    let newlabel = Array1::from(fmtclass);

    // Run through supervised training method
    let sw = Stopwatch::start_new();
    map.train_random_supervised(newdat, newlabel, 10000);
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
