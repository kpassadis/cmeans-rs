use core::f64;
use faer::Mat;
use std::print;

use serde::{Serialize, de::DeserializeOwned};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

/// Read a CSV file into a matrix. If a line contains an invalid entry (one that cannot be parsed into a floating point number)
/// the line will be completely skipped. If the CSV file contains a header the function will still work and it will merely ignore the
/// line.
pub fn load_csv_to_mat(
    filename: &str,
    skip_col: Option<usize>,
) -> Result<Mat<f64>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for line in reader.lines() {
        let row = line?;
        let row: Vec<Result<f64, _>> = row
            .split(",")
            .enumerate()
            .filter(|(i, _)| match skip_col {
                Some(skip) => *i != skip,
                None => true,
            })
            .map(|(_, s)| s.trim().parse())
            .collect();
        if row.iter().all(|el| el.is_ok()) {
            let row: Vec<f64> = row.into_iter().map(|el| el.unwrap()).collect();
            rows.push(row);
        }
    }

    let nrows = rows.len();

    if nrows == 0 {
        Err("No valid numeric rows found, check file validity".into())
    } else {
        let ncols = rows[0].len();
        let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);
        Ok(mat)
    }
}

// Saves any serializable model or struct to a nicely formatted JSON file on disk.
///
/// Uses a `BufWriter` for high-performance I/O, which is crucial when saving
/// models with large matrices.
pub fn save_model_to_json<T: Serialize, P: AsRef<Path>>(
    model: &T,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Using `to_writer_pretty` makes the JSON human-readable.
    // If file size is a strict concern, switch to `serde_json::to_writer`
    serde_json::to_writer_pretty(writer, model)?;

    Ok(())
}

/// Loads a model or struct from a JSON file on disk.
///
/// The generic type `T` must implement `DeserializeOwned` to guarantee that
/// the returned struct takes full ownership of its memory allocation, rather
/// than trying to borrow from the file stream.
pub fn load_model_from_json<T: DeserializeOwned, P: AsRef<Path>>(
    path: P,
) -> Result<T, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let model = serde_json::from_reader(reader)?;

    Ok(model)
}

#[macro_export]
macro_rules! map {
        ($key:ty, $val:ty) => {
            let map:HashMap<$key, $val> = HashMap::new();
            map
        };

        ($($key:expr => $val:expr), *) => {
            {
                let mut map = HashMap::new();
                $( map.insert($key, $val); )*
                map
            }
        };

        ($keys:expr, $key:ty, $val:ty) => {
            {
                let mut map:HashMap<$key, Vec<$val>> = HashMap::new();
                for key in $keys {
                    map.insert(key, Vec::new());
                }
                map
            }
        };

        ($keys:expr, $values:expr, $key:ty, $val:ty) => {
            {
                let mut map:HashMap<$key, Vec<$val>> = HashMap::new();
                $keys.iter().zip($values.iter()).for_each(|(key, value)| {
                    map.insert(*key, value.clone());
                });
                map
            }
        };
    }

pub(crate) use map;

pub enum Axis {
    Horizontal,
    Vertical,
}

pub enum Cmp {
    Max,
    Min,
}

pub fn submat(input: &Mat<f64>, idx: &[usize], axis: Axis) -> Mat<f64> {
    let (n, p) = input.shape();
    match axis {
        //Select rows
        Axis::Horizontal => {
            let n = idx.len();
            let mut mat: Mat<f64> = Mat::zeros(n, p);
            for i in 0..n {
                for j in 0..p {
                    mat[(i, j)] = input[(idx[i], j)];
                }
            }
            mat
        }
        //Select columns
        Axis::Vertical => {
            let p = idx.len();
            let mut mat: Mat<f64> = Mat::zeros(n, p);
            for i in 0..n {
                for j in 0..p {
                    mat[(i, j)] = input[(i, idx[j])];
                }
            }
            mat
        }
    }
}

pub fn sum(input: &Mat<f64>, axis: Axis) -> Mat<f64> {
    match axis {
        Axis::Vertical => {
            let mut result = Mat::<f64>::zeros(1, input.ncols());
            input.col_iter().enumerate().for_each(|(i, col)| {
                result[(0, i)] = col.sum();
            });
            result
        }
        Axis::Horizontal => {
            let mut result = Mat::<f64>::zeros(input.nrows(), 1);
            input.row_iter().enumerate().for_each(|(i, row)| {
                result[(i, 0)] = row.sum();
            });
            result
        }
    }
}

pub fn which(input: &Mat<f64>, axis: Axis, cmp: Cmp) -> Vec<usize> {
    match axis {
        Axis::Vertical => input
            .col_iter()
            .map(|col| {
                col.iter().enumerate().fold(0, |idx, (i, x)| match cmp {
                    Cmp::Max => {
                        if col[idx] > *x {
                            idx
                        } else {
                            i
                        }
                    }
                    Cmp::Min => {
                        if col[idx] < *x {
                            idx
                        } else {
                            i
                        }
                    }
                })
            })
            .collect::<Vec<usize>>(),
        _ => input
            .row_iter()
            .map(|col| {
                col.iter().enumerate().fold(0, |idx, (i, x)| match cmp {
                    Cmp::Max => {
                        if col[idx] > *x {
                            idx
                        } else {
                            i
                        }
                    }
                    Cmp::Min => {
                        if col[idx] < *x {
                            idx
                        } else {
                            i
                        }
                    }
                })
            })
            .collect::<Vec<usize>>(),
    }
}

/// A utility function to visually inspect the weights of a cluster.
/// Returns a histogram-like plot where each bar corresponds to a feature of the
/// dataset and the height of the bar represents the weight of the particular feature, which
/// is an indicator of the importance of the particular feature in the formation of the cluster.
pub fn print_membership_matrix(mat: &Mat<f64>, idx: usize) -> String {
    let p = mat.shape().1;
    let mut res: Vec<String> = Vec::new();
    let step = 1.0 / p as f64;
    for i in 0..p {
        let mut row: Vec<&str> = Vec::new();
        for j in 0..p {
            if mat[(idx, j)] - ((i + 1) as f64) * step > 0.0 {
                row.push("*");
            } else {
                row.push(".");
            }
        }
        res.push(row.join(""));
    }

    res.reverse();
    res.join("\n")
}

/// Calculates the squared Euclidean distance between two vectors of floats.
/// Accepts anything that can iterate over `&f64` (Rows, Cols, Slices, Vecs).
pub fn euclidean_distance_squared<'a, I>(x: I, y: I) -> f64
where
    I: IntoIterator<Item = &'a f64>,
{
    x.into_iter()
        .zip(y.into_iter())
        .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
}

/// Calculates the distance between a vector and all rows or columns in a matrix.
/// The function assumes that both matrices supplied are row oriented or column oriented.
/// In either case it will return a column matrix.
pub fn squared_distance_between(x: &Mat<f64>, y: &Mat<f64>, axis: Axis) -> Mat<f64> {
    match axis {
        Axis::Horizontal => {
            let x_vec = x.as_ref().get_r(0);
            Mat::from_fn(y.nrows(), 1, |i, _| {
                euclidean_distance_squared(y.as_ref().get_r(i).iter(), x_vec.iter())
            })
        }
        Axis::Vertical => {
            let x_vec = x.as_ref().get_c(0);
            Mat::from_fn(y.ncols(), 1, |i, _| {
                euclidean_distance_squared(y.as_ref().get_c(i).iter(), x_vec.iter())
            })
        }
    }
}

/// A utility function to calulate the distance matrix.
/// The input is a matrix of vectors the distances of which we want to calculate and
/// the axis indicates whether the vectors are stacked horizontally (one below the other) or vertically (one next to the other).
/// The distance matrix returned is a rectangualr matrix
pub fn distance_matrix(input: &Mat<f64>, axis: Axis) -> Mat<f64> {
    match axis {
        // Vectors stacked one below the other
        Axis::Horizontal => {
            let n = input.nrows();
            let mut distance_matrix = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                // Calculate only the lower triangle, the distance matrix is symmetric
                for j in 0..i {
                    let x = input.as_ref().get_r(i);
                    let y = input.as_ref().get_r(j);
                    let distance = euclidean_distance_squared(x.iter(), y.iter());
                    distance_matrix[(i, j)] = distance.sqrt();
                    distance_matrix[(j, i)] = distance.sqrt();
                }
            }
            distance_matrix
        }
        // Vectors stacked one next to the other
        Axis::Vertical => {
            let n = input.ncols();
            let mut distance_matrix = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..i {
                    let x = input.as_ref().get_c(i);
                    let y = input.as_ref().get_c(j);
                    let distance = euclidean_distance_squared(x.iter(), y.iter());
                    distance_matrix[(i, j)] = distance.sqrt();
                    distance_matrix[(j, i)] = distance.sqrt();
                }
            }
            distance_matrix
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{assert_eq, println};

    use crate::utils::submat;

    use super::{Axis, Cmp, distance_matrix, load_csv_to_mat, sum, which};
    use faer::{Mat, mat};

    #[test]
    fn test_distance_matrix() {
        let x = mat![
            [0.1, 0.15, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.21, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let dist_mat = distance_matrix(&x, Axis::Horizontal);
        assert_eq!(dist_mat.nrows(), 6);
        let dist_mat = distance_matrix(&x, Axis::Vertical);
        assert_eq!(dist_mat.nrows(), 3);
        println!("{:?}", &dist_mat);
    }

    #[test]
    fn test_which_horizontal_min() {
        let x = mat![
            [0.1, 0.15, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.21, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let res = which(&x, Axis::Horizontal, Cmp::Min);
        assert_eq!(res, vec![0, 0, 0, 1, 2, 0])
    }

    #[test]
    fn test_which_horizontal_max() {
        let x = mat![
            [0.1, 0.15, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.21, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let res = which(&x, Axis::Horizontal, Cmp::Max);
        assert_eq!(res, vec![2, 2, 2, 2, 1, 2])
    }

    #[test]
    fn test_sum() {
        let x = mat![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];

        //Summing acros the vertical axis returns a single row matrix with as many columns
        //as the input matrix
        let row_sum = sum(&x, Axis::Vertical);
        assert_eq!(row_sum.shape(), (1, 3));
        //Summing across the horizontal axis returns a matrix with a single column vector
        //with as many rows as the input matrix
        let col_sum = sum(&x, Axis::Horizontal);
        assert_eq!(col_sum.shape(), (2, 1));
    }

    #[test]
    fn test_submat() {
        let x = mat![
            [0.1, 0.15, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.21, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let x2 = submat(&x, &[1, 2], Axis::Horizontal);
        assert_eq!(x2.shape().0, 2);
    }

    #[test]
    fn test_load_csv_file_to_matrix() {
        let raw_data: Result<Mat<f64>, _> =
            load_csv_to_mat("/home/kpassadis/datasets/genes/data.csv", Some(0));
        assert!(raw_data.is_ok());
        let raw_data = raw_data.unwrap();
        println!("Row data len: {}", raw_data.ncols());
    }
}
