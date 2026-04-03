use cmeans::preprocessing::StandardScaler;
use cmeans::subspace::SubspaceKMeans;
use cmeans::utils;
use csv::ReaderBuilder;
use faer::Mat;
use itertools::Itertools;
use std::collections::HashMap;
use std::path::Path;

fn load_breast_cancer<P: AsRef<Path>>(path: P) -> (Mat<f64>, Mat<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap();

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    rdr.records().for_each(|r| {
        let r = r.unwrap();
        let mut x_row: Vec<f64> = Vec::new();
        r.iter().enumerate().for_each(|(j, x)| {
            if j == 1 {
                let value = if x == "M" { 1.0 } else { 0.0 };
                y.push(value);
            }
            if j > 1 {
                let value: f64 = x.parse().unwrap();
                x_row.push(value);
            }
        });
        x.push(x_row);
    });

    let n = x.len();
    let p = x[0].len();
    let mut x_mat = Mat::<f64>::zeros(n, p);
    let mut y_mat = Mat::<f64>::zeros(n, 1);

    for i in 0..n {
        y_mat[(i, 0)] = y[i];
        for j in 0..p {
            x_mat[(i, j)] = x[i][j];
        }
    }

    (x_mat, y_mat)
}

fn load_wine<P: AsRef<Path>>(path: P) -> (Mat<f64>, Mat<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap();

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    rdr.records().for_each(|r| {
        let r = r.unwrap();
        let mut x_row: Vec<f64> = Vec::new();
        r.iter().enumerate().for_each(|(j, x)| {
            let value: f64 = x.parse().unwrap();
            if j == 0 {
                y.push(value);
            } else {
                x_row.push(value);
            }
        });
        x.push(x_row);
    });

    let n = x.len();
    let p = x[0].len();
    let mut x_mat = Mat::<f64>::zeros(n, p);
    let mut y_mat = Mat::<f64>::zeros(n, 1);

    for i in 0..n {
        y_mat[(i, 0)] = y[i];
        for j in 0..p {
            x_mat[(i, j)] = x[i][j];
        }
    }

    (x_mat, y_mat)
}

fn load_faithful<P: AsRef<Path>>(path: P) -> Mat<f64> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap();

    let array = rdr
        .records()
        .map(|r| {
            let r = r.unwrap();
            r.iter()
                .map(|x| {
                    let x: f64 = x.parse().unwrap();
                    x
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let n = array.len();
    let p = array[0].len();
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = array[i][j];
        }
    }

    x
}

fn load_iris<P: AsRef<Path>>(path: P) -> (Mat<f64>, HashMap<String, Vec<usize>>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    let input_array: Vec<Vec<f64>> = rdr
        .records()
        .enumerate()
        .map(|(i, r)| {
            let r = r.unwrap();
            let r: Vec<f64> = r
                .iter()
                .map(|x| {
                    let f: Result<f64, _> = x.parse();
                    (i, f, x)
                })
                .map(|(i, r, s)| {
                    if r.is_err() {
                        let indexes = groups.get_mut(s);
                        if let Some(indexes) = indexes {
                            indexes.push(i);
                        } else {
                            groups.insert(s.to_owned(), vec![i]);
                        }
                    }
                    r
                })
                .filter(|x| x.is_ok())
                .map(|x| x.unwrap())
                .collect();
            r
        })
        .collect();

    let n = input_array.len();
    let p = input_array[0].len();
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = input_array[i][j];
        }
    }

    (x, groups)
}

#[test]
fn load_iris_returns_array() {
    let (x, groups) = load_iris("tests/data/iris.csv");
    assert_eq!(x.nrows(), 150);
    assert_eq!(x.ncols(), 4);
    assert_eq!(groups.keys().len(), 3);
    println!("{:?}", groups);
}

#[test]
fn iris_data_is_clustered() {
    let (x, _) = load_iris("tests/data/iris.csv");
    let model =
        cmeans::FuzzyMeans::fit(3, 2.0, &x, 150, cmeans::UpdateMethod::ImmediateCenterUpdate);
    let memberships = model.predict_memberships(&x);
    assert_eq!(memberships.ncols(), 3);
    let preds = model.predict_hard(&x);
    // Check each class chunk (50 samples) has a dominant cluster
    for (chunk_idx, chunk) in preds.chunks(50).enumerate() {
        let counts = chunk.iter().counts();
        let (&_label, &max) = counts.iter().max_by_key(|(_, v)| *v).unwrap();

        assert!(
            max >= 30,
            "chunk {chunk_idx} not clustered: counts={counts:?}"
        );
    }
}

#[test]
fn test_faithful() {
    let c = 10;
    let x = load_faithful("tests/data/faithful_geyser.csv");
    let model =
        cmeans::FuzzyMeans::fit(c, 2.0, &x, 150, cmeans::UpdateMethod::ImmediateCenterUpdate);
    let memberships = model.predict_memberships(&x);
    assert_eq!(memberships.ncols(), c);
    let eps = 1e-4;
    for row in memberships.row_iter() {
        let sum = row.sum();
        assert!((1.0 - eps <= sum) && (sum <= 1.0 + eps));
    }

    //println!("{:?}", model.evaluate(&x));
}

#[test]
fn test_wine() {
    let (x, y) = load_wine("tests/data/wine.csv");
    assert_eq!(x.shape().0, 178);
    assert_eq!(x.shape().0, y.shape().0);

    let scaler = StandardScaler::fit(&x);
    let x_scaled = scaler.transform(&x);

    let model = SubspaceKMeans::fit(3, 2.0, 1e-6, &x_scaled, 30);
    println!("{:?}", model.get_progress());
    println!("{:?}", model.get_members());
    println!("{:?}", model.get_weights());
}

#[test]
fn test_breast_cancer_subspace() {
    let (x, y) = load_breast_cancer("tests/data/breast_cancer.csv");
    assert_eq!(x.shape().0, y.shape().0);

    let scaler = StandardScaler::fit(&x);
    let x_scaled = scaler.transform(&x);

    let model = SubspaceKMeans::fit(2, 2.0, 1e-6, &x_scaled, 30);
    println!("{:?}", model.get_members());
    println!("{:?}", model.get_weights());

    let weights = model.get_weights();
    let progress = model.get_progress();
    println!("{}", utils::print_membership_matrix(weights, 0));
}

#[test]
fn test_breast_cancer_cmeans() {
    let (x, y) = load_breast_cancer("tests/data/breast_cancer.csv");
    assert_eq!(x.shape().0, y.shape().0);

    let scaler = StandardScaler::fit(&x);
    let x_scaled = scaler.transform(&x);
    let model = cmeans::FuzzyMeans::fit(
        2,
        2.0,
        &x_scaled,
        20,
        cmeans::UpdateMethod::ImmediateCenterUpdate,
    );
    let memberships = model.predict_memberships(&x_scaled);
    let hard = model.predict_hard(&x_scaled);
    println!("{:?}", hard);
    println!("{:?}", model.get_history());
}
