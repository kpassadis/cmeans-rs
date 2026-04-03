use crate::utils::{Axis, Cmp, submat, sum, which};
use faer::Mat;
use rand::prelude::*;

/// Hard subspace clustering model based on the Subspace K-Means variant.
///
/// The model stores both the hyperparameters used for training and the learned
/// clustering state after fitting.
///
/// # Fields
///
/// - `c`: Number of clusters.
/// - `alpha`: Subspace weighting exponent. Must be greater than `1.0`.
/// - `epsilon`: Small positive constant used for numerical stability.
/// - `weights`: Learned feature-weight matrix of shape `(c, p)`, where `p` is
///    the number of input features. Each row contains the feature weights of one cluster.
/// - `centers`: Learned cluster-center matrix of shape `(c, p)`.
/// - `members`: Hard cluster assignments for the training points. If present,
///   `members[i]` is the cluster index assigned to input point `i`, so each value belongs to `{0, ..., c - 1}`.
/// - `progress`: Objective-function value at each training iteration.
pub struct SubspaceKMeans {
    c: usize,
    alpha: f64,
    epsilon: f64,
    weights: Option<Mat<f64>>,
    centers: Option<Mat<f64>>,
    members: Option<Vec<usize>>,
    progress: Vec<f64>,
}

impl SubspaceKMeans {
    /// Internal constructor used in the fit function to initialize the hyperparameters of a model.
    fn new(c: usize, alpha: f64, epsilon: f64) -> Self {
        let epsilon = epsilon.max(1e-6);
        let alpha = alpha.max(1.5);
        Self {
            c,
            alpha,
            epsilon,
            weights: None,
            centers: None,
            members: None,
            progress: vec![],
        }
    }

    /// Returns a reference to the the scores that the objective value that were recorded
    /// during the training process.
    pub fn get_progress<'a>(&'a self) -> &'a Vec<f64> {
        let progress = self.progress.as_ref();
        progress
    }

    /// Returns a reference to the weight matrix.
    pub fn get_weights<'a>(&'a self) -> &'a Mat<f64> {
        let weights = self
            .weights
            .as_ref()
            .expect("model has not been initialized");
        weights
    }

    /// Returns a reference to an index vector where each position of the vector is assigned a discrete value in the set {0,...c-1}
    pub fn get_members(&self) -> &[usize] {
        self.members
            .as_ref()
            .expect("model has not been initialized")
    }

    /// Initializes the model state from the input data.
    ///
    /// This method initializes:
    /// - the cluster-specific weight matrix,
    /// - the cluster-center matrix,
    /// - and the initial hard membership assignments.
    ///
    /// Both the weight matrix and the center matrix have shape `(c, p)`, where
    /// `c` is the number of clusters and `p` is the number of input features.
    ///
    /// Initialization proceeds as follows:
    /// - all feature weights are initialized uniformly to `1.0 / p`,
    /// - `c` input samples are selected uniformly at random as the initial cluster centers,
    /// - memberships are assigned using the initial centers and weights,
    /// - centers are updated from the initial memberships,
    /// - weights are updated from the initial memberships and centers.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`, where `n` is the number of samples.
    ///
    /// # Panics
    ///
    /// Panics if `self.c > n`.
    fn init(&mut self, x: &Mat<f64>) {
        let (n, p) = x.shape();
        assert!(self.c <= n, "c must be <= n");
        let mut weights = Mat::<f64>::ones(self.c, p);
        let mut centers = Mat::<f64>::zeros(self.c, p);

        for i in 0..self.c {
            for j in 0..p {
                weights[(i, j)] /= p as f64;
            }
        }

        let mut rng = rand::rng();
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);

        indexes.iter().take(self.c).enumerate().for_each(|(i, r)| {
            for j in 0..p {
                centers[(i, j)] = x[(*r, j)];
            }
        });
        self.weights = Some(weights);
        self.centers = Some(centers);
        self.update_members(x);
        self.update_centers(x);
        self.update_weights(x);
    }

    /// Updates the hard cluster assignment of each input sample.
    ///
    /// For each sample, this method computes the weighted distance to every cluster
    /// center and assigns the sample to the cluster with minimum distance.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`, where `n` is the number of samples
    ///   and `p` is the number of features.
    fn update_members(&mut self, x: &Mat<f64>) {
        let weights = self
            .weights
            .as_ref()
            .expect("model has not been initialized");
        let distances = self.weighted_distances(x, weights);
        let members = which(&distances, Axis::Vertical, Cmp::Min);
        self.members = Some(members);
    }

    /// Computes the weighted squared Euclidean distance matrix.
    ///
    /// For cluster `k` and sample `i`, the returned matrix contains
    ///
    /// `d[k, i] = sum_j w[k, j]^alpha * (x[i, j] - center[k, j])^2`.
    ///
    /// The returned matrix has shape `(c, n)`, where `c` is the number of clusters
    /// and `n` is the number of input samples.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`.
    /// - `weights`: Cluster-specific feature-weight matrix of shape `(c, p)`.
    ///
    /// # Returns
    ///
    /// A distance matrix of shape `(c, n)`.
    ///
    /// # Panics
    ///
    /// Panics if the model centers have not been initialized.
    pub fn weighted_distances(&self, x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
        let (n, p) = x.shape();
        debug_assert_eq!(weights.shape(), (self.c, p));
        debug_assert_eq!(self.centers.as_ref().unwrap().shape(), (self.c, p));
        //Distance matrix has shape [kxn]
        let mut distances = Mat::<f64>::zeros(self.c, n);
        //Get a reference to the cluster centers. Note that the init method must have been called
        //otherwise unwrap will cause panic.
        let centers = self
            .centers
            .as_ref()
            .expect("Init method must have been called");
        //In the loop we calculate the distances for each cluster.
        distances
            .row_iter_mut()
            .enumerate()
            .for_each(|(k, mut row)| {
                let center = centers.row(k);
                let w_k = weights.row(k);
                // precompute w^alpha
                let mut walpha = vec![0.0; p];
                for j in 0..p {
                    walpha[j] = w_k[j].powf(self.alpha);
                }

                for i in 0..n {
                    let mut dist = 0.0;
                    for j in 0..p {
                        let d = x[(i, j)] - center[j];
                        dist += walpha[j] * d * d;
                    }
                    row[i] = dist;
                }
            });

        distances
    }

    /// Updates the cluster-specific feature-weight matrix.
    ///
    /// For each cluster, this method computes the within-cluster dispersion of each
    /// feature and updates the corresponding weight vector so that features with
    /// smaller within-cluster variance receive larger weights.
    ///
    /// The updated weights are normalized so that each cluster weight vector sums to `1`.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`.
    ///
    /// # Panics    
    ///
    /// Panics if the model has not been initialized.
    fn update_weights(&mut self, x: &Mat<f64>) {
        let (_, p) = x.shape();
        //Cloning the weights is not expensive as the dimensions are [c x p], which should not be large
        //even for a moderately large number of features.
        let mut weights = self
            .weights
            .clone()
            .expect("Init method must have been called");
        let centers = self.centers.as_ref().unwrap();
        let members = self.members.as_ref().unwrap();
        //clusters is k vectors (one per cluster) and each contains the indexes of
        //the clusters memebrs of the training data points.
        let mut clusters = vec![Vec::new(); self.c];
        for (i, &k) in members.iter().enumerate() {
            clusters[k].push(i);
        }

        //Compute the power outside the loop so that you don't have to recompute
        let power = -1.0 / (self.alpha - 1.0);

        //println!("Members: {:?}", members);
        //Each row of the weight corresponds to the weights of a cluster (there are c of those).
        weights
            .row_iter_mut()
            .zip(centers.row_iter())
            .enumerate()
            .for_each(|(k, (mut wv, center))| {
                //Get the rows of the dataset that has been assigned to the particular cluster
                let cluster_members = &clusters[k];

                //println!("{k}, {:?}", &cluster_members);
                if !cluster_members.is_empty() {
                    let mut dists = Mat::<f64>::zeros(cluster_members.len(), p);
                    cluster_members.iter().enumerate().for_each(|(i, k)| {
                        for j in 0..p {
                            dists[(i, j)] = (x[(*k, j)] - center[j]).powi(2);
                        }
                    });

                    //Step 2: Sum across the vertical axis to get a matrix of dimension [1xp]
                    //Step 3: calculate the total sum of distances
                    //Step 4: divide each distance by the total. This is the new weight vector
                    let dists = sum(&dists, Axis::Vertical);
                    dists.col_iter().enumerate().for_each(|(i, col)| {
                        wv[i] = col[0] + self.epsilon;
                        wv[i] = wv[i].powf(power);
                    });

                    let total = wv.sum();
                    if total > 0.0 && total.is_finite() {
                        wv.iter_mut().for_each(|x| {
                            *x /= total;
                        });
                    } else {
                        wv.iter_mut().for_each(|x| {
                            *x = 1.0 / p as f64;
                        });
                    }
                }
            });

        self.weights = Some(weights);
    }

    /// Updates the cluster centers of the model.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`, where `n` is the number of samples
    ///   and `p` is the number of features.
    fn update_centers(&mut self, x: &Mat<f64>) {
        let mut centers = self.centers.clone().expect("Init must be called");
        let members = self.members.as_ref().unwrap();
        let mut clusters = vec![Vec::new(); self.c];
        for (i, &k) in members.iter().enumerate() {
            clusters[k].push(i);
        }
        centers
            .row_iter_mut()
            .enumerate()
            .for_each(|(k, mut center)| {
                let cluster_members = &clusters[k];
                if !cluster_members.is_empty() {
                    //Update the cluster center.
                    //The submat function returns the subset of x that belongs to this cluster
                    //This implementation is suboptimal since the submat creates copies
                    let xm = submat(x, cluster_members, Axis::Horizontal);
                    //Sum the elements of the matrix across the vertical axis.
                    let xm = sum(&xm, Axis::Vertical);
                    xm.col_iter().enumerate().for_each(|(j, c)| {
                        center[j] = c[0] / cluster_members.len() as f64;
                    });
                }
            });

        self.centers = Some(centers);
    }

    /// Computes the value of the clustering objective under the current model state.
    ///
    /// The objective is the sum, over all input samples, of the weighted squared
    /// Euclidean distance to the cluster currently assigned to each sample.
    ///
    /// This value is useful for:
    /// - monitoring training progress,
    /// - checking convergence,
    /// - and comparing different runs of the algorithm.
    ///
    /// In a correctly implemented alternating optimization scheme, the objective
    /// should decrease monotonically up to numerical tolerance.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`, where `n` is the number of samples
    ///   and `p` is the number of features.
    ///
    /// # Returns
    ///
    /// The objective value given a cluster configuration.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been initialized.
    pub fn objective(&self, x: &Mat<f64>) -> f64 {
        let weights = self.weights.as_ref().expect("init() must be called");
        let d = self.weighted_distances(x, weights); // [c x n]
        let members = self.members.as_ref().expect("members must be set");

        let mut obj = 0.0;
        for (i, &k) in members.iter().enumerate() {
            obj += d[(k, i)];
        }
        obj
    }

    /// Fits the model parameters to the input data.
    ///
    /// The input data matrix `x` has shape `(n, p)`, where `n` is the number of
    /// samples and `p` is the number of features.
    ///
    /// Training proceeds by alternating between three steps:
    /// 1. update the hard cluster assignment of each sample,
    /// 2. update the cluster centers,
    /// 3. update the cluster-specific feature weights.
    ///
    /// The procedure stops when either:
    /// - no sample changes cluster assignment, or
    /// - the maximum number of iterations `n_iter` is reached.
    ///
    /// # Arguments
    ///
    /// - `x`: Input data matrix of shape `(n, p)`.
    /// - `n_iter`: Maximum number of training iterations.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been initialized.
    fn fit_model(&mut self, x: &Mat<f64>, n_iter: usize) {
        assert!(
            self.centers.is_some() && self.weights.is_some(),
            "Call init() before fit()"
        );
        let mut progress: Vec<f64> = Vec::new();
        for _ in 0..n_iter {
            let prev = self.members.clone();
            self.update_members(x);
            if prev.as_ref() == self.members.as_ref() {
                break; // no members changed cluster, converged
            }
            self.update_centers(x);
            self.update_weights(x);
            progress.push(self.objective(x));
        }

        self.progress = progress;
    }

    /// Fits a Subspace K-Means model to the input data and returns the trained model.
    ///
    /// This is the main entry point for training.
    ///
    /// # Arguments
    ///
    /// - `c`: Number of clusters.
    /// - `alpha`: Subspace weighting exponent. Must be greater than `1.0`.
    /// - `epsilon`: Small positive constant used for numerical stability.
    /// - `x`: Input data matrix of shape `(n, p)`.
    /// - `n_iter`: Maximum number of training iterations.
    ///
    /// # Returns
    ///
    /// A trained [`SubspaceKMeans`] model containing the learned cluster centers,
    /// feature weights, hard assignments, and objective-value history.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `c > n`, where `n` is the number of input samples,
    /// - or if `alpha <= 1.0`.
    ///
    /// # Notes
    ///
    /// For real-world datasets, it is usually advisable to standardize the input
    /// features before fitting the model. A preprocessing module is provided in the library that 
    /// enables feature scaling without resorting to a separate dependency. 
    pub fn fit(c: usize, alpha: f64, epsilon: f64, x: &Mat<f64>, n_iter: usize) -> Self {
        let mut model = SubspaceKMeans::new(c, alpha, epsilon);
        model.init(x);
        model.fit_model(&x, n_iter);
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_weighted_distances_shape() {
        let x = mat![
            [0.1, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];

        let c: usize = 3;
        let alpha = 1.0;
        let epsilon = 1e-8;
        let mut model = SubspaceKMeans::new(c, alpha, epsilon);
        model.init(&x);

        let weights = Mat::<f64>::ones(c, 3);
        let distances = model.weighted_distances(&x, &weights);
        assert_eq!(distances.nrows(), c);
        assert_eq!(distances.ncols(), x.nrows());
    }

    #[test]
    fn test_init() {
        let x = mat![
            [0.1, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
            [0.123, 0.11, 0.9],
            [0.42, 0.122, 0.5],
        ];

        let c: usize = 2;
        let alpha = 2.0;
        let epsilon = 1e-8;
        let mut model = SubspaceKMeans::new(c, alpha, epsilon);
        model.init(&x);

        model.update_weights(&x);
        println!("{:?}", model.weights.as_ref().unwrap());
        assert_eq!(model.weights.unwrap().sum() as usize, 2)
    }
}
