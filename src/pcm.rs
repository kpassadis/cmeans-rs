use faer::{Mat, RowRef, unzip, zip};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    thread::{self, ScopedJoinHandle},
};

use crate::metrics::ClusterMetrics;
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
    gamma:f64
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
        let mut t: Mat<f64> = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            t[(i, 0)] = 1.0 / (c as f64);
        }
        for i in 0..p {
            v[(0, i)] = row[i];
        }
        Self { v, t, m, gamma:0.0 }
    }

    /// The contribution of the cluster to the total objective. The distances matrix
    /// constains the distance of each point in the dataset from the center of the 
    /// particular cluster.
    /// # Arguments
    /// - `distances`: A reference to a distance matrix of shape [n x 1]. 
    /// 
    fn objective(&self, distances:&Mat<f64>) -> f64 {
        assert_eq!(self.t.shape(), distances.shape());
        let mut obj = 0.0;
        zip!(&self.t, distances).for_each(|unzip!(t, d)| {
            obj += t.powf(self.m) * d;
        });
        
        obj
    }

    fn penalty(&self) -> f64 {
        todo!()
    }
}