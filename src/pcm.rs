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
    t: Mat<f64>,
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
        let t = Mat::from_fn(1, p, |_, j| row[j]);
        Self {
            v,
            t,
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
        assert_eq!(self.t.shape(), distances.shape());
        let mut obj = 0.0;
        zip!(&self.t, distances).for_each(|unzip!(t, d)| {
            obj += t.powf(self.m) * d;
        });

        obj
    }

    /// The objective function in the PCM has two terms. The second term
    /// is similar to a penalty term that prevents the first term from setting
    /// the t for each pattern and cluster to zero.
    fn term2(&self) -> f64 {
        let penalty = self
            .t
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

        zip!(&mut self.t, &dist_sq).for_each(|unzip!(t, d)| {
            *t = if *d == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + (d / gamma).powf(p))
            };
        });
    }
}
