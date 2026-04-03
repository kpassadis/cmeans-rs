use core::f64;
use faer::Mat;

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

#[cfg(test)]
mod tests {

    use crate::utils::submat;

    use super::{Axis, Cmp, sum, which};
    use faer::mat;

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
}
