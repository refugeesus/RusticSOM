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
    pub pos: String,
    pub challengee: String,
    pub witness_gateway: String,
    pub signal: f64,
    pub snr: f64,
    pub fspl: f64,
    pub label: u32,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct TestSamplesTag {
    pub pos: String,
    pub challengee: String,
    pub witness_gateway: String,
    pub signal: f64,
    pub snr: f64,
    pub fspl: f64,
    pub label: u32,
    pub slabel: String,
}

fn normalize_sig(rssi: f64, snr: f64, fspl: f64) -> [f64; 3] {
    let rout = (rssi - (1.0 * 10f64.powi(-17))) / (0.001 - (1.0 * 10f64.powi(-17)));
    //let rout = (rssi - (-134.0)) / (0.0 - (-134.0));
    let sout = (snr - (-19.0)) / (17.0 - (-19.0));
    let fout = (fspl - (3.981072 * 10f64.powi(-20))) / (0.001 - (3.981072 * 10f64.powi(-20)));
    //let fout = (fspl - (-164.0)) / (0.0 - (-164.0));
    [rout, sout, fout]
    //[rssi, snr, fspl]
}

#[test]
fn test_unsupervised() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            signal: record.signal,
            snr: record.snr,
            fspl: record.fspl,
            label: record.label,
            slabel: match record.label {
                1 => "real".to_string(),
                0 => "fake".to_string(),
                _ => "undefined".to_string(),
            },
        };
        samples.push(rec_tag);
    }

    // Format parsed training samples
    let testsamplen = samples.len();
    let mut fmtdata: Vec<[f64; 3]> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();
    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(normalize_sig(data.signal, data.snr, data.fspl));
        fmtclass.push(data.slabel.clone());
    }

    // Initialize classes if there are any
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("real".to_string(), 0.0);
    classes.insert("fake".to_string(), 0.0);

    // Create a new SOM using default settings
    let mut map = SOM::create(
        15,
        15,
        3,
        true,
        Some(0.5),
        Some(0.75),
        None,
        None,
        Some(classes),
        None,
        Some([209,162,182,84,44,167,62,240,152,122,118,154,48,208,143,84,
            186,211,219,113,71,108,171,185,51,159,124,176,167,192,23,245])
    );
    let newdat = Array2::from(fmtdata);
    let newdat2 = newdat.clone();

    // Unsupervised training of the SOM
    let sw = Stopwatch::start_new();
    map.train_random(newdat, 1600);
    println!("TrainUnsupervised: {:?}", sw.elapsed());

    // Output relative distance map of SOM nodes
    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    // Write trained SOM results to file for plotting externally
    let mut file = File::create("outputs/output_unsupervised_15_3.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    // Write out condensed winner list
    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}

#[test]
fn test_supervised() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            signal: record.signal,
            snr: record.snr,
            fspl: record.fspl,
            label: record.label,
            slabel: match record.label {
                1 => "real".to_string(),
                0 => "fake".to_string(),
                _ => "undefined".to_string(),
            },
        };
        samples.push(rec_tag);
    }

    let testsamplen = samples.len();
    let mut fmtdata: Vec<[f64; 3]> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();
    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(normalize_sig(data.signal, data.snr, data.fspl));
        fmtclass.push(data.slabel.clone());
    }
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("real".to_string(), 0.0);
    classes.insert("fake".to_string(), 0.0);
    let classes = classes.clone();

    let mut map = SOM::create(
        15,
        15,
        3,
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
    let newdat = Array2::from(fmtdata);
    let newdat2 = newdat.clone();
    let newlabel = Array1::from(fmtclass);

    // Run through supervised training method
    let sw = Stopwatch::start_new();
    map.train_random_supervised(newdat, newlabel, 2000);
    println!("TrainSupervised: {:?}", sw.elapsed());

    let mut file = File::create("outputs/output_supervised_15_3.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}


#[test]
fn test_unsupervised_10_3() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            signal: record.signal,
            snr: record.snr,
            fspl: record.fspl,
            label: record.label,
            slabel: match record.label {
                1 => "real".to_string(),
                0 => "fake".to_string(),
                _ => "undefined".to_string(),
            },
        };
        samples.push(rec_tag);
    }

    // Format parsed training samples
    let testsamplen = samples.len();
    let mut fmtdata: Vec<[f64; 3]> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();
    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(normalize_sig(data.signal, data.snr, data.fspl));
        fmtclass.push(data.slabel.clone());
    }

    // Initialize classes if there are any
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("real".to_string(), 0.0);
    classes.insert("fake".to_string(), 0.0);

    // Create a new SOM using default settings
    let mut map = SOM::create(
        10,
        10,
        3,
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
    let newdat = Array2::from(fmtdata);
    let newdat2 = newdat.clone();

    // Unsupervised training of the SOM
    let sw = Stopwatch::start_new();
    map.train_random(newdat, 1600);
    println!("TrainUnsupervised: {:?}", sw.elapsed());

    // Output relative distance map of SOM nodes
    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    // Write trained SOM results to file for plotting externally
    let mut file = File::create("outputs/output_unsupervised_10_3.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    // Write out condensed winner list
    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}

#[test]
fn test_supervised_10_3() -> Result<(), Box<dyn std::error::Error>> {
    // Read training samples from file and parse into structure for consumption
    let mut samples: Vec<TestSamplesTag> = Vec::new();
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/aggregate_samples.csv")?;
    let iter = rdr.into_deserialize();
    for result in iter {
        let record: TestSamples = result.unwrap();
        let rec_tag: TestSamplesTag = TestSamplesTag {
            pos: record.pos,
            challengee: record.challengee,
            witness_gateway: record.witness_gateway,
            signal: record.signal,
            snr: record.snr,
            fspl: record.fspl,
            label: record.label,
            slabel: match record.label {
                1 => "real".to_string(),
                0 => "fake".to_string(),
                _ => "undefined".to_string(),
            },
        };
        samples.push(rec_tag);
    }

    let testsamplen = samples.len();
    let mut fmtdata: Vec<[f64; 3]> = Vec::new();
    let mut fmtclass: Vec<String> = Vec::new();
    for i in 0..testsamplen {
        let data = samples.get(i).unwrap();
        fmtdata.push(normalize_sig(data.signal, data.snr, data.fspl));
        fmtclass.push(data.slabel.clone());
    }
    let mut classes: HashMap<String, f64> = HashMap::new();
    classes.insert("real".to_string(), 0.0);
    classes.insert("fake".to_string(), 0.0);
    let classes = classes.clone();

    let mut map = SOM::create(
        10,
        10,
        3,
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
    let newdat = Array2::from(fmtdata);
    let newdat2 = newdat.clone();
    let newlabel = Array1::from(fmtclass);

    // Run through supervised training method
    let sw = Stopwatch::start_new();
    map.train_random_supervised(newdat, newlabel, 2000);
    println!("TrainSupervised: {:?}", sw.elapsed());

    let mut file = File::create("outputs/output_supervised_10_3.json")?;
    file.write_all(map.to_json()?.as_bytes())?;

    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    for x in newdat2.genrows() {
        let y = x.to_owned();
        let _winner = map.winner(y);
    }

    Ok(())
}
