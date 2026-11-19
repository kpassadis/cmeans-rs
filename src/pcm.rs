use faer::{Mat, RowRef, unzip, zip};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    thread::{self, ScopedJoinHandle},
};

use crate::metrics::{self, ClusterMetrics};
use crate::serialization::mat_serde;

/// Represents a single possibilistic cluster within the Possibilistic C-Means model.
///
/// A cluster maintains its centroid and a membership matrix that tracks
/// how strongly each data point in the training set belongs to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cluster {
    /// The cluster center. A matrix of dimensions `[1 x p]` (a row vector), where `p` is
    /// the dimensionality of the input space.
    #[serde(with = "mat_serde")]
    v: Mat<f64>,
    /// The typicality column-vector. Dimensions `[n x 1]`, where `n` is the number of data points.
    /// Each value represents the degree of compatibility of the corresponding point to this cluster.
    /// Typicality differs from membership in that it is an unnormalized score indicating the possibility
    /// of the point belonging to this particular cluster, independently of all other clusters.
    #[serde(with = "mat_serde")]
    tau: Mat<f64>,
    /// The fuzzifier hyperparameter `m`. Controls the "fuzziness" of the cluster assignments.
    m: f64,
    gamma: f64,
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
        assert!(
            m > 0.0,
            "the value of m specified should be larger than zero"
        );
        let v = Mat::from_fn(n, 1, |_, _| 1.0 / (c as f64));
        let tau = Mat::from_fn(1, p, |_, j| row[j]);
        Self {
            v,
            tau,
            m,
            gamma: 0.0,
        }
    }

    fn distances(&self, x: &Mat<f64>) -> Mat<f64> {
        metrics::sq_euclidean_distances(self.v.row(0), x.as_ref())
    }

    /// The contribution of the cluster to the total objective. The distances matrix
    /// constains the distance of each point in the dataset from the center of the
    /// particular cluster.
    /// # Arguments
    /// - `distances`: A reference to a distance matrix of shape [n x 1].
    /// Note that distances refers to a squared distance.
    fn term1(&self, distances: &Mat<f64>) -> f64 {
        assert_eq!(self.tau.shape(), distances.shape());
        let mut obj = 0.0;
        zip!(&self.tau, distances).for_each(|unzip!(t, d)| {
            obj += t.powf(self.m) * d;
        });

        obj
    }

    /// The objective function in the PCM has two terms. The second term
    /// is similar to a penalty term that prevents the first term from setting
    /// the t for each pattern and cluster to zero.
    fn term2(&self) -> f64 {
        let penalty = self
            .tau
            .row_iter()
            .fold(0.0, |acc, row| acc + (1.0 - row[0]).powf(self.m));
        self.gamma * penalty
    }

    /// Updates the typicality column-vector `t` for this cluster given the input data `x`.
    ///
    /// Computes the squared Euclidean distance from every sample in `x` to the cluster center `v`
    /// and updates each sample's typicality according to the Possibilistic C-Means stationarity
    /// condition:
    ///
    /// $$t_{ij} = \frac{1}{1 + \left( \frac{d_{ij}^2}{\gamma_i} \right)^{\frac{1}{m - 1}}}$$
    ///
    /// # Arguments
    /// - `x`: Input data matrix of dimensions `[n x p]`, where `n` is the number of samples
    ///   and `p` is the dimensionality of the feature space.
    fn update_tau(&mut self, x: &Mat<f64>) {
        let dist_sq = metrics::sq_euclidean_distances(self.v.row(0), x.as_ref());
        let p = 1.0 / (self.m - 1.0);
        let gamma = self.gamma.max(f64::EPSILON);

        zip!(&mut self.tau, &dist_sq).for_each(|unzip!(t, d)| {
            *t = if *d == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + (d / gamma).powf(p))
            };
        });
    }

    /// Updates the cluster center `v` as a typicality-weighted average of the input samples `x`.
    ///
    /// Computes the new prototype position according to the Possibilistic C-Means stationarity
    /// condition:
    ///
    /// $$\mathbf{v}_i = \frac{\sum_{j=1}^N t_{ij}^m \, \mathbf{x}_j}{\sum_{j=1}^N t_{ij}^m}$$
    ///
    /// If the total typicality mass $\sum_{j=1}^N t_{ij}^m$ collapses below machine epsilon
    /// (for instance, if all samples fall far outside the cluster bandwidth $\gamma_i$), the
    /// existing prototype `self.v` is preserved to prevent `NaN` propagation.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of dimensions `[n x p]`, where `n` is the number of samples
    ///   and `p` is the dimensionality of the feature space.
    fn update_v(&mut self, x: &Mat<f64>) {
        // Compute the [1 x n] row vector of weights w_j = tau_j^m and their sum
        let mut denom = 0.0;
        let weights = Mat::from_fn(1, x.nrows(), |_, i| {
            let w = self.tau[(i, 0)].powf(self.m);
            denom += w;
            w
        });

        // Guard against cluster collapse (all typicalities -> 0)
        if denom <= f64::EPSILON {
            return;
        }

        // Single SIMD matrix multiplication: [1 x n] * [n x p] -> [1 x p], then scale by 1/denom
        let mut v = &weights * x;
        let inv_denom = 1.0 / denom;
        zip!(&mut v).for_each(|unzip!(val)| *val *= inv_denom);

        self.v = v;
    }

    /// Updates the cluster bandwidth (scale) parameter `gamma` from the current typicalities `t`.
    ///
    /// Re-estimates the cluster dispersion scale according to the adaptive Possibilistic
    /// C-Means (PCM-II) heuristic:
    ///
    /// $$\gamma_i = \frac{\sum_{j=1}^N t_{ij}^m \, d_{ij}^2}{\sum_{j=1}^N t_{ij}^m}$$
    ///
    /// If the total typicality mass $\sum_{j=1}^N t_{ij}^m$ collapses below machine epsilon,
    /// the existing `self.gamma` value is preserved; otherwise, `self.gamma` is clamped to a
    /// small positive floor (`1e-12`) to prevent division-by-zero in subsequent typicality updates.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of dimensions `[n x p]`, where `n` is the number of samples
    ///   and `p` is the dimensionality of the feature space.
    fn update_gamma(&mut self, x: &Mat<f64>) {
        let dist_sq = metrics::sq_euclidean_distances(self.v.row(0), x.as_ref());

        let mut num = 0.0;
        let mut den = 0.0;
        zip!(&self.tau, &dist_sq).for_each(|unzip!(t, d)| {
            let t_m = t.powf(self.m);
            num += t_m * d;
            den += t_m;
        });

        if den > f64::EPSILON {
            self.gamma = (num / den).max(1e-12);
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PossibilisticMeans {
    clusters: Vec<Cluster>,
    history: Vec<f64>,
}

impl PossibilisticMeans {

    pub fn fit(c: usize, m: f64, x: &Mat<f64>, n_iter: usize) -> Self {
        assert!(m > 1.05);
        let p = x.ncols();
        let n = x.nrows();
        assert!(n > c);

        // Initialize prototypes and bandwidths (gamma_i) from a short FCM pass
        let fcm = crate::FuzzyMeans::fit(c, m, x, 15, crate::UpdateMethod::BatchCenterUpdate);

        let mut clusters: Vec<Cluster> = fcm
            .clusters
            .iter()
            .map(|fcm_cl| {
                let mut cl = Cluster::new(p, m, &fcm_cl.v.row(0), n, c);
                let dist_sq = crate::metrics::sq_euclidean_distances(cl.v.row(0), x.as_ref());

                // Estimate initial gamma_i from FCM memberships u_ij^m
                let mut num = 0.0;
                let mut den = 0.0;
                for j in 0..n {
                    let u_m = fcm_cl.mu[(j, 0)].powf(m);
                    num += u_m * dist_sq[(j, 0)];
                    den += u_m;
                }
                cl.gamma = if den > f64::EPSILON {
                    (num / den).max(f64::EPSILON)
                } else {
                    1.0
                };
                cl
            })
            .collect();

        let mut history: Vec<f64> = Vec::with_capacity(n_iter);

        // Coupled PCM-II alternating updates
        for _ in 0..n_iter {
            let mut max_center_shift = 0.0_f64;
            let mut obj = 0.0;

            for cl in clusters.iter_mut() {
                let prev_v = cl.v.clone();

                cl.update_tau(x);
                cl.update_gamma(x);
                cl.update_v(x);

                let dist_sq = crate::metrics::sq_euclidean_distances(cl.v.row(0), x.as_ref());
                obj += cl.term1(&dist_sq) + cl.term2();

                // Measure prototype shift for convergence
                let shift = (&cl.v - &prev_v).norm_l2();
                if shift > max_center_shift {
                    max_center_shift = shift;
                }
            }

            history.push(obj);

            if max_center_shift < 1e-6 {
                break;
            }
        }

        Self { clusters, history }
    }
}
