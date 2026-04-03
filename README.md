# cmeans-rs

A high-performance Rust crate for soft clustering and subspace clustering algorithms.

## Fuzzy C-Means

The fuzzy c-means algorithm performs soft clustering, where each data point can belong to multiple clusters with a varying degree of membership. It minimizes the following objective:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?J_m%20=%20\sum_{j=0}^{k-1}\sum_{i=0}^{n-1}u_{ij}^m%20D_{mh}(x_i,\mu_j)^2" alt="Fuzzy C-Means Objective" />
</div>

where:
* `u_ij` is the membership of the `i^{th}` data vector to cluster `j`.
* `m` is the fuzzifier (weight exponent) that controls cluster fuzziness.
* `\mu_j` is the center of cluster `j`.
* `D_{mh}` is the Mahalanobis (or standardized Euclidean) distance.

## Subspace Clustering

In subspace clustering, each attribute has a degree of membership associated with each cluster, indicating the importance of that feature to the cluster's formation. This algorithm is incredibly powerful for datasets with a large number of numeric features. 

For example, in a wine dataset with 13 numeric features, the algorithm learns feature weights to best assign a quality class label. In a breast cancer dataset with 60 features, it learns the optimal feature subspaces to distinguish between benign and malignant tumors.

The Subspace clustering algorithm minimizes the following objective:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?E_{\alpha,\epsilon}%20=%20\sum_{j=0}^{k-1}\sum_{x%20\in%20C_j}\sum_{r=0}^{d-1}w_{jr}^\alpha(x_r%20-%20\mu_{jr})^2%20+%20\epsilon%20\sum_{j=0}^{k-1}\sum_{r=0}^{d-1}w_{jr}^\alpha" alt="Subspace Clustering Objective" />
</div>

where:
* `\alpha \in (1, \infty)` is a weight component or fuzzifier.
* `\epsilon` is a very small positive constant for numerical stability. 
* `w_{jr} \in [0,1]` is an entry in the weight matrix. 

The equation to update the feature weights is given by:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?w_{jr}%20=%20\frac{(\sum_{x%20\in%20C_j}(x_r%20-%20\mu_{jr})^2%20+%20\epsilon)^{\frac{-1}{\alpha-1}}}{\sum_{i=0}^{d-1}(\sum_{x%20\in%20C_j}(x_i%20-%20\mu_{ji})^2%20+%20\epsilon)^{\frac{-1}{\alpha-1}}}" alt="Weight Update" />
</div>

And the equation to update the cluster centers is given by:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mu_{jr}%20=%20\frac{\sum_{x%20\in%20C_j}%20x_r}{|C_j|}" alt="Center Update" />
</div>

## Example Usage

```rust
use cmeans_rs::preprocessing::StandardScaler;
use cmeans_rs::subspace::SubspaceKMeans;
use cmeans_rs::utils;

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