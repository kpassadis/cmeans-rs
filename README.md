# cmeans-rs

A high-performance Rust crate for soft clustering and subspace clustering algorithms.

## fuzzy c-means

The fuzzy cmeans algorithm minimizes the following objective:

$$
J_{\alpha} = \sum_{j=0}^{k-1}\sum_{i=0}^{n-1}u_{ij}^{\alpha}D_{mh}(x_i,\mu_{j})^2
$$

where:

- $u_{ij}$ is the membership of the $i^{th}$ data vector to cluster j.
- $\mu_{j}$ is the center of cluster j.
- $D_{mh}$ is the Mahalanobi distance.

## Subspace Clustering

In subspace clustering, each attribute has a degree of membership associated with each cluster, indicating the importance of that feature to the cluster's formation. This algorithm is incredibly powerful for datasets with a large number of numeric features.

For example, in a wine dataset with 13 numeric features, the algorithm learns feature weights to best assign a quality class label. In a breast cancer dataset with 60 features, it learns the optimal feature subspaces to distinguish between benign and malignant tumors.

The Subspace clustering algorithm minimizes the following objective:

$$
E_{\alpha,\epsilon} = \sum_{j=0}^{k-1}\sum_{x \in C}\sum_{r=0}^{d-1}w_{jr}^a(x_r - \mu_{jr})^2 + \epsilon \sum_{j=0}^{k-1}\sum_{r=0}^{d-1}w_{jr}^\alpha
$$

where:

- $\alpha \in (1, \inf)$ is a weight component or fuzzifier.
- $\epsilon$ is a very small positive number. 
- $w_{jr} \in [0,1]$ is an entry in the weight matrix. 

The equation to update the weights is given by:

$$
w_{jr} = \frac{(\sum_{x \in C_j}(x_r - \mu_{jr})^2 + \epsilon)^{\frac{-1}{\alpha-1}}}{\sum_{i=0}^{d-1}(\sum_{x \in C_j}(x_i - \mu_{ji})^2 + \epsilon)^{\frac{-1}{\alpha-1}}}
$$

And the equation to update the cluster centers is given by:

$\mu_{jr} = \frac{\sum_{x \in C_{j}x_r}}{|C_j|}$

```rust

//Subspace k means example
let (x, y) = load_breast_cancer("tests/data/breast_cancer.csv");

let scaler = StandardScaler::fit(&x);
let x_scaled = scaler.transform(&x);
let model = SubspaceKMeans::fit(2, 2.0, 1e-6, &x_scaled, 30);
let weights = model.get_weights();
let progress = model.get_progress();
println!("{}", utils::print_membership_matrix(weights, 0));

```


## TODO

- [ ] Async/parallel support.
- [ ] Serde support.
- [ ] Plotting utilities.
- [ ] Synthetic data generation.
- [ ] Hot loop optimization.

## License

- MIT license ([LICENSE-MIT](LICENSE-MIT))