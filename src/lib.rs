use faer::{Mat, RowRef, unzip, zip};
use rand::prelude::*;
use std::{
    collections::HashMap,
    thread::{self, ScopedJoinHandle},
};

pub mod metrics;
pub mod preprocessing;
pub mod subspace;
pub mod utils;

use crate::metrics::ClusterMetrics;

/// Represents a single fuzzy cluster within the Fuzzy C-Means model.
///
/// A cluster maintains its centroid and a membership matrix that tracks
/// how strongly each data point in the training set belongs to it.
#[derive(Debug, Clone)]
struct Cluster {
    /// The cluster center. A matrix of dimensions `[1 x p]`, where `p` is 
    /// the dimensionality of the input space.
    v: Mat<f64>,
    /// The membership column-vector. Dimensions `[n x 1]`, where `n` is the number of data points.
    /// Each value represents the degree of membership of the corresponding point to this cluster.
    mu: Mat<f64>,
    /// The fuzzifier hyperparameter `m`. Controls the "fuzziness" of the cluster assignments.
    m: f64,
}

/// Defines the strategy used to update cluster centers during the Expectation-Maximization loop.
#[derive(Clone)]
pub enum UpdateMethod {
    /// Centers are updated sequentially, immediately after their corresponding memberships are updated.
    ImmediateCenterUpdate,
    /// Memberships for all clusters are updated first, followed by a batch update of all centers.
    BatchCenterUpdate,
}

impl Cluster {
    /// Instantiates a new cluster.
    ///
    /// # Arguments
    /// - `p`: Dimensionality of the input space (number of features).
    /// - `m`: The fuzzifier hyperparameter.
    /// - `row`: A reference to a row from the input data matrix to initialize the center.
    /// - `n`: Number of samples in the input data.
    /// - `c`: Total number of clusters in the model.
    fn new(p: usize, m: f64, row: &RowRef<f64>, n: usize, c: usize) -> Self {
        let mut v: Mat<f64> = Mat::<f64>::zeros(1, p);
        let mut mu: Mat<f64> = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            mu[(i, 0)] = 1.0 / (c as f64);
        }
        for i in 0..p {
            v[(0, i)] = row[i];
        }
        Self { v, mu, m }
    }

    /// Calculates the squared standardized distance between the cluster center and each data point.
    /// Uses a diagonal weight matrix `G` to scale the distances.
    /// Given a matrix G [1 x p], an input matrix X [N x p] calculate the
    /// distances between the center of the cluster and each row of the matrix.
    /// This returns a vector of length N, which is equal to the number of patterns.
    /// It turns out that it will be convenient to return the distances as a column
    /// matrix instead of arranging in rows. The reason is that it will be
    /// much more efficient to perform the calculation of the membership step.
    /// Returns a distance matrix of shape [Nx1], where the ith element of the matrix
    /// is the distance between the cluster center and the corresponding ith pattern.
    /// # Returns
    /// A column matrix of shape `[n x 1]` containing the distances.
    fn distances(&self, x: &Mat<f64>, g: &Mat<f64>) -> Mat<f64> {
        assert_eq!(self.v.nrows(), 1);
        assert_eq!(g.nrows(), 1);
        assert_eq!(g.ncols(), self.v.ncols());

        let p = self.v.ncols();
        let n = x.nrows();
        let mut d = Mat::<f64>::zeros(1, p);
        //The column matrix to return
        let mut dist_mat = Mat::<f64>::zeros(n, 1);
        x.row_iter().enumerate().for_each(|(i, x)| {
            let x = x.as_mat();
            zip!(&mut d, &self.v, g, &x)
                .for_each(|unzip!(d, v, g, x)| *d = (*x - *v) * *g * (*x - *v));
            let dsum = d.sum();
            dist_mat[(i, 0)] = dsum;
        });

        dist_mat
    }

    /// Computes the cluster's contribution to the total objective function.
    fn objective(&self, distances: &Mat<f64>, m: f64) -> f64 {
        assert_eq!(self.mu.shape(), distances.shape());
        let mut obj = 0.0;
        zip!(&self.mu, distances).for_each(|unzip!(mu, d)| {
            obj += mu.powf(m) * d;
        });
        obj
    }

    /// Updates the membership matrix (`mu`) based on the current distances.
    fn update_mu(&mut self, cluster_dists: &Mat<f64>, sums_of_inv_dists: &Mat<f64>) {
        let alpha = 1.0 / (self.m - 1.0);
        let eps = 1e-6;
        zip!(&mut self.mu, cluster_dists, sums_of_inv_dists).for_each(
            |unzip!(mu_ik, d_ik, denom_i)| {
                let w_ik = (1.0 / (*d_ik + eps)).powf(alpha);
                *mu_ik = w_ik / *denom_i;
            },
        );
    }

    //// Updates the cluster center (`v`) based on the current membership degrees.
    fn update_v(&mut self, x: &Mat<f64>) {
        //The denominator is simply the summ of all membership values raised to m.
        let denom: f64 = self.mu.row_iter().map(|row| row[0].powf(self.m)).sum();

        //The nominator is calculated in two steps:
        //1. Multiply each training example with its corresponding membership value.
        //   you can think of as assigning an importance of a particular example to the
        //   formulation of the cluster center.
        //2. Then we sum all the examples across the dimensions. And finally we divide
        //   the resulting vector with the denominator we calculated earlier.

        let mut v: Mat<f64> = Mat::zeros(1, self.v.ncols());

        //Broadcast the membership column vector into a matrix with p columns.
        //This requires space allocation but allows us to do faster math using faer.
        //The reason is that we get a matrix that has the same dimensionality as the
        //input matrix, but the memberships are repeated across the columns for each row.
        let mut v_temp: Mat<f64> =
            Mat::from_fn(x.nrows(), x.ncols(), |i, _| self.mu[(i, 0)].powf(self.m));
        zip!(&mut v_temp, x).for_each(|unzip!(v, x)| *v = *v * *x);
        v_temp.col_iter().enumerate().for_each(|(i, col)| {
            let x: f64 = col.sum();
            let x = x / denom;
            v[(0, i)] = x;
        });
        self.v = v;
    }
}

pub struct FuzzyMeans {
    clusters: Vec<Cluster>,
    g: Mat<f64>,
    history: Vec<f64>,
}

impl FuzzyMeans {
    /// Calculates `G`, a matrix of dimensions `[1 x p]`.
    ///
    /// This represents the inverse variance of each feature. It acts as the diagonal of a 
    /// Mahalanobis-style covariance matrix, standardizing the distance calculation across 
    /// features of different scales.
    fn g(x: &Mat<f64>) -> Mat<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let eps = 1e-12;
        let n_inv = 1.0 / (n - 1) as f64;
        let inv_variances = x
            .col_iter()
            .map(|c| {
                let column_mean = c.sum() / (n as f64);
                let column_variance: f64 = c
                    .iter()
                    .map(|xi| n_inv * (xi - column_mean) * (xi - column_mean))
                    .sum();
                1.0 / (eps + column_variance)
            })
            .collect::<Vec<f64>>();

        let g = Mat::<f64>::from_fn(1, p, |_, j| inv_variances[j]);
        g
    }

    /// Predicts the fuzzy membership degrees of new test data.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    ///
    /// # Returns
    /// A membership matrix of shape `[n x c]`, where each row sums to `1.0`. The value at `(i, j)`
    /// represents the degree to which pattern `i` belongs to cluster `j`..
    pub fn predict_memberships(&self, x: &Mat<f64>) -> Mat<f64> {
        let m = self.clusters[0].m;
        let alpha = 1.0 / (m - 1.0);
        let eps = 1e-6;
        let n = x.nrows();
        let c = self.clusters.len();

        let mut distances: Vec<Mat<f64>> = Vec::with_capacity(c);

        self.clusters.iter().for_each(|c| {
            let cluster_dists = c.distances(x, &self.g);
            let mut inv_distances = Mat::<f64>::zeros(cluster_dists.nrows(), cluster_dists.ncols());
            zip!(&mut inv_distances, &cluster_dists)
                .for_each(|unzip!(id, d)| *id = f64::powf(1.0 / (*d + eps), alpha));
            distances.push(inv_distances);
        });
        //Have to sum across all clusters. This basically means that
        //i have to sum the each row of elements that spans across all elements of the vector
        //Better place these in a matrix instead.
        (0..n).for_each(|i| {
            let sum = distances.iter().fold(0.0, |acc, m| acc + m[(i, 0)]);
            let sum = if sum.is_finite() && sum > 0.0 {
                sum
            } else {
                1.0
            };
            distances.iter_mut().for_each(|m| {
                m[(i, 0)] /= sum;
            });
        });

        let membership_matrix = Mat::from_fn(distances[0].nrows(), distances.len(), |i, j| {
            distances[j][(i, 0)]
        });

        membership_matrix
    }

    /// Converts fuzzy memberships into crisp/hard cluster assignments.
    ///
    /// Assigns each data point to the cluster for which it has the highest membership degree.
    ///
    /// # Returns
    /// A vector of length `n` containing the assigned cluster index `[0, c-1]` for each data point.
    pub fn predict_hard(&self, x: &Mat<f64>) -> Vec<usize> {
        let memberships = self.predict_memberships(x);
        utils::which(&memberships, utils::Axis::Horizontal, utils::Cmp::Max)
    }

    /// Fits a Fuzzy C-Means model to the input data.
    ///
    /// # Arguments
    /// - `c`: The target number of clusters. Must be less than `n`.
    /// - `m`: The fuzzifier coefficient. Must be strictly greater than `1.05` (typically between `1.5` and `2.5`).
    /// - `x`: The input training matrix of shape `[n x p]`.
    /// - `n_iter`: The maximum number of training iterations.
    /// - `method`: The sequence used to update centers (`Immediate` or `Batch`).
    ///
    /// # Returns
    /// A trained `FuzzyMeans` model.
    ///
    /// # Panics
    /// Panics if `m <= 1.05` or if the number of samples `n` is less than or equal to `c`.
    pub fn fit(c: usize, m: f64, x: &Mat<f64>, n_iter: usize, method: UpdateMethod) -> Self {
        assert!(m > 1.05);
        let eps = 1e-6;
        let p = x.ncols();
        let n = x.nrows();
        assert!(n > c);

        let mut rng = rand::rng();
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);

        let mut clusters: Vec<Cluster> = Vec::new();
        //Initialize the clusters
        indexes.into_iter().take(c).for_each(|i| {
            let xi = x.row(i);
            let cluster = Cluster::new(p, m, &xi, n, c);
            clusters.push(cluster);
        });

        //calculate the matrix G.
        let g = Self::g(x);
        let alpha = 1.0 / (m - 1.0);

        let mut history: Vec<f64> = Vec::new();
        //Update clusters for a number of iterations
        for _ in 0..n_iter {
            let mut sums_of_inv_dists = Mat::<f64>::zeros(n, 1);
            let mut distances: Vec<Mat<f64>> = Vec::new();
            let mut obj = 0.0;
            clusters.iter().for_each(|c| {
                let cluster_dists = c.distances(x, &g);
                zip!(&mut sums_of_inv_dists, &cluster_dists).for_each(|unzip!(t, d)| {
                    *t = *t + f64::powf(1.0 / (*d + eps), alpha);
                });
                obj += c.objective(&cluster_dists, m);
                distances.push(cluster_dists);
            });

            match method {
                UpdateMethod::ImmediateCenterUpdate => {
                    clusters.iter_mut().enumerate().for_each(|(i, c)| {
                        //Step 2: Update the memberships.
                        c.update_mu(&distances[i], &sums_of_inv_dists);
                        //Step 3: Update the cluster centers.
                        c.update_v(x);
                    });
                }
                UpdateMethod::BatchCenterUpdate => {
                    for (i, cl) in clusters.iter_mut().enumerate() {
                        cl.update_mu(&distances[i], &sums_of_inv_dists);
                    }

                    // 3. update all centers
                    for cl in clusters.iter_mut() {
                        cl.update_v(x);
                    }
                }
            }

            history.push(obj);

            if history.len() >= 2 {
                let last = history[history.len() - 1];
                let prev = history[history.len() - 2];
                if ((prev - last).abs() / prev.max(1.0)) < 1e-8 {
                    break;
                }
            }
        }

        FuzzyMeans {
            clusters,
            g,
            history,
        }
    }

    /// Returns a slice containing the objective function values recorded
    /// at the end of each training iteration.
    pub fn get_history<'a>(&'a self) -> &'a Vec<f64> {
        &self.history
    }

    /// Performs a parallelized random grid search to tune hyper-parameters `c` and `m`.
    ///
    /// Evaluates combinations of clusters `c` and fuzzifier `m` across a random subset of
    /// configurations. This process is multithreaded using standard scoped threads to 
    /// ensure high performance. Models are evaluated based on their entropy coefficient.
    ///
    /// # Arguments
    /// - `x`: Input data matrix.
    /// - `min_c`: Minimum number of clusters to test.
    /// - `max_c`: Maximum number of clusters to test.
    /// - `n`: Number of random parameter combinations to evaluate.
    ///
    /// # Returns
    /// A vector of `TuneResult` containing the hyper-parameters and resulting metric.
    pub fn tune_rgs(x: &Mat<f64>, min_c: usize, max_c: usize, n: usize) -> Vec<TuneResult> {
        let min_m = 1.5;
        let max_m = 3.0;
        let step_m = 0.1;

        assert!(min_c < max_c);
        assert!(
            max_c - min_c > 5,
            "Max and min values of c should differ at least by 5"
        );

        let mus = (0..16)
            .into_iter()
            .map(|i| min_m + step_m * i as f64)
            .take_while(|x| *x <= max_m);

        let mut rng = rand::rng();
        let cs = (0..100)
            .into_iter()
            .map(|i| min_c + i)
            .take_while(|c| *c <= max_c);

        let mut combinations = mus
            .map(|mu| {
                let mut cs = cs.clone().map(|c| (mu, c)).collect::<Vec<(f64, usize)>>();
                cs.shuffle(&mut rng);
                cs
            })
            .collect::<Vec<Vec<(f64, usize)>>>();
        //Shuffle again to to outer index
        combinations.shuffle(&mut rng);

        let combinations: Vec<&Vec<(f64, usize)>> = combinations.iter().take(n).collect();
        let chunk_size = combinations.len() / 4;

        let res = thread::scope(|scope| {
            let mut results: Vec<TuneResult> = Vec::new();
            let combinations = combinations.chunks(chunk_size);
            let handlers = combinations
                .map(|v| {
                    scope.spawn(move || {
                        let v = v
                            .to_vec()
                            .iter()
                            .map(|v| v.iter())
                            .flatten()
                            .map(|(m, c)| {
                                let model = FuzzyMeans::fit(
                                    *c,
                                    *m,
                                    x,
                                    100,
                                    UpdateMethod::ImmediateCenterUpdate,
                                );
                                let u = model.predict_memberships(x);
                                let entropy = FuzzyMeans::entropy_coefficient(&u);
                                //let partition_coef = FuzzyMeans::partition_coefficient(&u);
                                TuneResult::new(*c, *m, entropy, "entropy".to_string())
                            })
                            .collect::<Vec<TuneResult>>();
                        v
                    })
                })
                .collect::<Vec<ScopedJoinHandle<Vec<TuneResult>>>>();
            for h in handlers {
                if let Ok(mut v) = h.join() {
                    results.append(&mut v);
                }
            }
            results
        });

        res
    }
}

pub struct TuneResult {
    /// The evaluated number of clusters.
    pub c: usize,
    /// The evaluated fuzzifier coefficient.
    pub m: f64,
    /// The resulting metric value (e.g., entropy).
    pub metric: f64,
    /// The name of the metric used for evaluation.
    pub metric_name: String,
}

/// Stores the result of a single hyper-parameter tuning iteration.
impl TuneResult {
    pub fn new(c: usize, m: f64, metric: f64, metric_name: String) -> Self {
        Self {
            c,
            m,
            metric,
            metric_name,
        }
    }
}

impl ClusterMetrics for FuzzyMeans {
    /// Computes the partition coefficient of the membership matrix.
    ///
    /// Measures the amount of overlap between clusters. Values range from `1/c` (maximum fuzziness) 
    /// to `1.0` (hard clustering).
    fn partition_coefficient(u: &Mat<f64>) -> f64 {
        let n = u.nrows() as f64;
        let mut sum_sq = 0.0;
        for row in u.row_iter() {
            for &v in row.iter() {
                sum_sq += v * v;
            }
        }

        sum_sq / n
    }

    /// Computes the partition entropy of the membership matrix.
    ///
    /// Measures the fuzziness of the cluster assignments. Lower values indicate crisper partitions.
    fn entropy_coefficient(u: &Mat<f64>) -> f64 {
        let n = u.nrows() as f64;
        let eps = 1e-12;
        let mut sum = 0.0;
        for row in u.row_iter() {
            for &v in row.iter() {
                let x = v.max(eps);
                sum += -x * x.log2();
            }
        }

        sum / n
    }

    fn evaluate(&self, x: &Mat<f64>) -> HashMap<String, f64> {
        let u = self.predict_memberships(x);
        let entropy = Self::entropy_coefficient(&u);
        let partition_coef = Self::partition_coefficient(&u);

        let mut map = HashMap::new();
        map.insert("entropy".to_owned(), entropy);
        map.insert("partition_coefficient".to_owned(), partition_coef);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_which_returns_correct_indices() {
        let x = mat![
            [0.1, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let indices = utils::which(&x, utils::Axis::Horizontal, utils::Cmp::Max);
        assert!(indices == vec![2, 2, 2, 2, 1, 2]);
        let indices = utils::which(&x, utils::Axis::Horizontal, utils::Cmp::Min);
        assert!(indices == vec![1, 0, 1, 1, 2, 0]);
        let indices = utils::which(&x, utils::Axis::Vertical, utils::Cmp::Max);
        assert!(indices == vec![4, 4, 5]);
        let indices = utils::which(&x, utils::Axis::Vertical, utils::Cmp::Min);
        assert!(indices == vec![5, 0, 0]);
    }

    #[test]
    fn predicted_memberships_add_up_to_one() {
        let x = mat![
            [0.1, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let fuzzy = FuzzyMeans::fit(2, 2.0, &x, 10, UpdateMethod::ImmediateCenterUpdate);
        let u = fuzzy.predict_memberships(&x);

        let eps = 1e-4;

        for i in 0..u.nrows() {
            let s: f64 = u.row(i).iter().sum();
            assert!(s.is_finite(), "row {i} sum not finite: {s}");
            assert!((s - 1.0).abs() < eps, "row {i} sum = {s}, expected ~1.0");
        }
    }
}
