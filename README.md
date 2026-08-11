# cmeans-rs

A high-performance Rust crate for soft clustering and subspace clustering algorithms.

## Fuzzy C-Means

The fuzzy c-means algorithm performs soft clustering, where each data point can belong to multiple clusters with a varying degree of membership. It minimizes the following objective:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white%20J_m%20=%20\sum_{j=0}^{k-1}\sum_{i=0}^{n-1}u_{ij}^m%20D_{mh}(x_i,\mu_j)^2" alt="Fuzzy C-Means Objective" />
</div>

where:
* *u*<sub>ij</sub> is the membership of the *i*<sup>th</sup> data vector to cluster *j*.
* *m* is the fuzzifier (weight exponent) that controls cluster fuzziness.
* *μ*<sub>j</sub> is the center of cluster *j*.
* *D*<sub>mh</sub> is the Mahalanobis (or standardized Euclidean) distance.

## Subspace Clustering

In subspace clustering, each attribute has a degree of membership associated with each cluster, indicating the importance of that feature to the cluster's formation. This algorithm is incredibly powerful for datasets with a large number of numeric features. 

For example, in a wine dataset with 13 numeric features, the algorithm learns feature weights to best assign a quality class label. In a breast cancer dataset with 60 features, it learns the optimal feature subspaces to distinguish between benign and malignant tumors.

The Subspace clustering algorithm minimizes the following objective:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white%20E_{\alpha,\epsilon}%20=%20\sum_{j=0}^{k-1}\sum_{x%20\in%20C_j}\sum_{r=0}^{d-1}w_{jr}^\alpha(x_r%20-%20\mu_{jr})^2%20+%20\epsilon%20\sum_{j=0}^{k-1}\sum_{r=0}^{d-1}w_{jr}^\alpha" alt="Subspace Clustering Objective" />
</div>

where:
* *α* ∈ (1, ∞) is a weight component or fuzzifier.
* *ε* is a very small positive constant for numerical stability. 
* *w*<sub>jr</sub> ∈ [0, 1] is an entry in the weight matrix. 

The equation to update the feature weights is given by:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white%20w_{jr}%20=%20\frac{(\sum_{x%20\in%20C_j}(x_r%20-%20\mu_{jr})^2%20+%20\epsilon)^{\frac{-1}{\alpha-1}}}{\sum_{i=0}^{d-1}(\sum_{x%20\in%20C_j}(x_i%20-%20\mu_{ji})^2%20+%20\epsilon)^{\frac{-1}{\alpha-1}}}" alt="Weight Update" />
</div>

And the equation to update the cluster centers is given by:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white%20\mu_{jr}%20=%20\frac{\sum_{x%20\in%20C_j}%20x_r}{|C_j|}" alt="Center Update" />
</div>

## Example Usage

<u> Breast cancer dataset </u>

```rust
use cmeans::preprocessing::StandardScaler;
use cmeans::subspace::SubspaceKMeans;
use cmeans::utils;

fn main() {
    // 1. Load your data (implementation depends on your dataset)
    let (x, y) = load_breast_cancer("tests/data/breast_cancer.csv");
    assert_eq!(x.shape().0, y.shape().0);

    // 2. Scale the features (Highly recommended for distance-based clustering)
    let scaler = StandardScaler::fit(&x);
    let x_scaled = scaler.transform(&x);

    // 3. Fit the Subspace K-Means model
    // c = 2 clusters, alpha = 2.0, epsilon = 1e-6, max_iter = 30
    let model = SubspaceKMeans::fit(2, 2.0, 1e-6, &x_scaled, 30);

    // 4. Extract and visualize the learned feature weights
    let weights = model.get_weights();
    println!("Feature weights for Cluster 0:\n{}", utils::print_membership_matrix(weights, 0));
}

```
<u> Gene dataset </u>

Gene Expression Cancer RNA-Seq dataset on the UCI Machine Learning Repository is a random extraction from the larger PANCAN dataset. It contains gene expression data for 801 samples and 20,531 genes, categorized into five distinct tumor types.

```rust

use std::time::Instant;
use cmeans::preprocessing::{MinMaxScaler, StandardScaler, drop_zero_variance_columns, log2_transform};
use cmeans::utils::load_csv_to_mat;
use cmeans::subspace::SubspaceKMeans;
use faer::Mat;

//Drop the first column, it is non numeric
let raw_data: Mat<f64> = load_csv_to_mat("data.csv", Some(0))?;

let clean_data: Mat<f64> = drop_zero_variance_columns(&raw_data);
let log_data: Mat<f64> = log2_transform(&clean_data);

println!("Applying Min-Max scaling...");
let scaler: MinMaxScaler = MinMaxScaler::fit(&log_data); 
let x_scaled: Mat<f64> = scaler.transform(&log_data);

let clusters:usize = 5;
let beta:f64 = 5.1; 
let tolerance:f64 = 1e-10;
let max_iters:usize = 200;

println!("Starting Subspace K-Means...");
let start_time:Instant = Instant::now();
let model: SubspaceKMeans = SubspaceKMeans::fit(clusters, beta, tolerance, &x_scaled, max_iters);
let duration:std::time::Duration = start_time.elapsed(); 
println!("✅ Training completed in: {:?}", duration);
println!("{:?}", model.get_members());

```


## Hyperparameter Tuning & Tips

- **Subspace Exponent (`beta`)**: The `beta` parameter controls the strength of feature weighting. Recommended ranges for high-dimensional genomic/expression data fall between **3.3 and 5.3**. Higher values enforce stricter feature selection, while lower values behave closer to standard weighted K-Means.
- **Random Seeding**: Because Subspace K-Means optimization can occasionally land in local minima depending on initial centroid placement, running multiple initializations or testing slight variations in `beta` is recommended for optimal convergence.

🚀 What's New in v0.2.0

- Disk I/O & Serialization: Full serde support for seamlessly saving and loading trained models to and from disk.
- Cluster Variance Metrics: Calculate cluster variance and standard deviations.

🚀 What's New in v0.3.0

- Enhance preprocessing module with MinMax scaler and a function to drop zero variance columns from a matrix.
- Provide a utility function to read a CSV file into a faer matrix.

## Roadmap & Future Development

- [ ] **Multi-Start Initializations (`n_init`)**: Implement multiple random restarts to systematically bypass suboptimal local minima, automatically selecting and returning the model instance that minimizes the objective function $P(U, Z, W)$.
- [ ] **Unsupervised Internal Validation Metrics**: Add native support for label-free clustering evaluation metrics tailored for subspace structures (e.g., Weighted Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index).
- [ ] **Feature Weight Entropy & Sparsity Diagnostics**: Implement entropy tracking for the weight vector $W$ to detect and prevent uniform dimension weight collapse or noise hyper-concentration.