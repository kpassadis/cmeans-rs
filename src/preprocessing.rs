use faer::Mat;

/// The function will drop columns that have zero variance.
///
/// # Arguments
/// - `mat`: a reference to a feature matrix
///
/// # Returns
/// - A reconstructed matrix without the zero variance columns
pub fn drop_zero_variance_columns(mat: &Mat<f64>) -> Mat<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    let mut valid_columns = Vec::with_capacity(ncols);

    // Check each column to see if it fluctuates
    for j in 0..ncols {
        let mut is_constant = true;

        let first_val = mat[(0, j)];

        for i in 1..nrows {
            // Using a small epsilon to account for floating-point inaccuracies
            if (mat[(i, j)] - first_val).abs() > 1e-9 {
                is_constant = false;
                break;
            }
        }

        if !is_constant {
            valid_columns.push(j);
        }
    }

    let new_ncols = valid_columns.len();
    println!("Dropped {} zero-variance columns.", ncols - new_ncols);

    // Reconstruct a new matrix using only the valid columns
    Mat::from_fn(nrows, new_ncols, |i, j| mat[(i, valid_columns[j])])
}

/// Takes log2 transformation on the elements of the input matrix
///
pub fn log2_transform(mat: &Mat<f64>) -> Mat<f64> {
    let (nrows, ncols) = mat.shape();

    Mat::from_fn(nrows, ncols, |i, j| {
        // We add 1.0 to prevent taking log2(0), which is negative infinity
        (mat[(i, j)] + 1.0).log2()
    })
}

#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    x_min: Vec<f64>,
    x_max: Vec<f64>,
}

impl MinMaxScaler {
    /// Fits the scaler to the data by learning the minimum and maximum of each feature.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    pub fn fit(x: &Mat<f64>) -> Self {
        let (nrows, ncols) = x.shape();
        let mut x_min = vec![f64::MAX; ncols];
        let mut x_max = vec![f64::MIN; ncols];

        // Iterate through the matrix to find the min and max for each column
        for j in 0..ncols {
            for i in 0..nrows {
                let val = x[(i, j)];
                if val < x_min[j] {
                    x_min[j] = val;
                }
                if val > x_max[j] {
                    x_max[j] = val;
                }
            }
        }

        Self { x_min, x_max }
    }

    /// Transforms the data using the learned min and max.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    pub fn transform(&self, x: &Mat<f64>) -> Mat<f64> {
        let (nrows, ncols) = x.shape();

        Mat::from_fn(nrows, ncols, |i, j| {
            let range = self.x_max[j] - self.x_min[j];

            // 🚨 Crucial: Prevent division by zero for zero-variance features!
            if range == 0.0 {
                0.0 // If min == max, all values are identical. Map them to 0.0.
            } else {
                (x[(i, j)] - self.x_min[j]) / range
            }
        })
    }

    /// Fits the scaler to the data and then transforms it in a single step.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    pub fn fit_transform(x: &Mat<f64>) -> (Self, Mat<f64>) {
        let scaler = Self::fit(x);
        let transformed = scaler.transform(x);
        (scaler, transformed)
    }

    /// Scales the data back to its original representation.
    ///
    /// # Arguments
    /// - `x_scaled`: A previously scaled data matrix.
    pub fn inverse_transform(&self, x_scaled: &Mat<f64>) -> Mat<f64> {
        let (nrows, ncols) = x_scaled.shape();

        Mat::from_fn(nrows, ncols, |i, j| {
            let range = self.x_max[j] - self.x_min[j];
            x_scaled[(i, j)] * range + self.x_min[j]
        })
    }

    /// Returns the learned minimum of each feature.
    pub fn min(&self) -> &[f64] {
        &self.x_min
    }

    /// Returns the learned maximum of each feature.
    pub fn max(&self) -> &[f64] {
        &self.x_max
    }
}

/// Standardizes features by removing the mean and scaling to unit variance.
///
/// Centering and scaling happen independently on each feature by computing
/// the relevant statistics on the samples in the training set.
///
/// This preprocessing step is highly recommended before applying distance-based
/// clustering algorithms like Fuzzy C-Means or Subspace K-Means, as features
/// with larger scales can disproportionately dominate the distance calculations.
pub struct StandardScaler {
    x_mean: Vec<f64>,
    x_sd: Vec<f64>,
}

impl StandardScaler {
    /// Computes the mean and standard deviation for each feature in the dataset.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`, where `n` is the number of samples
    ///   and `p` is the number of features.
    ///
    /// # Returns
    /// A fitted `StandardScaler` instance containing the learned parameters.
    pub fn fit(x: &Mat<f64>) -> Self {
        let (n, _) = x.shape();
        let eps = 1e-10; // Threshold for zero variance

        let x_mean = x
            .col_iter()
            .map(|c| c.sum() / n as f64)
            .collect::<Vec<f64>>();

        let x_sd = x
            .col_iter()
            .enumerate()
            .map(|(i, c)| {
                let mu = x_mean[i];
                let total_dev: f64 = c.iter().map(|x| (x - mu).powi(2)).sum();
                let sd = f64::sqrt(total_dev / (n - 1) as f64);

                // Safety check: Prevent division by zero for constant features.
                // If variance is 0, we set the scale to 1.0 so the feature is merely centered.
                if sd < eps { 1.0 } else { sd }
            })
            .collect::<Vec<f64>>();

        Self { x_mean, x_sd }
    }

    /// Performs standard scaling on the input data using the fitted parameters.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    ///
    /// # Returns
    /// A new matrix of shape `[n x p]` where each feature has zero mean and unit variance.
    pub fn transform(&self, x: &Mat<f64>) -> Mat<f64> {
        let (nrows, ncols) = x.shape();

        // faer's from_fn is a very clean way to allocate and fill a matrix in one go
        Mat::from_fn(nrows, ncols, |i, j| {
            (x[(i, j)] - self.x_mean[j]) / self.x_sd[j]
        })
    }

    /// Fits the scaler to the data and then transforms it in a single step.
    ///
    /// # Arguments
    /// - `x`: Input data matrix of shape `[n x p]`.
    pub fn fit_transform(x: &Mat<f64>) -> (Self, Mat<f64>) {
        let scaler = Self::fit(x);
        let transformed = scaler.transform(x);
        (scaler, transformed)
    }

    /// Scales the data back to its original representation.
    ///
    /// # Arguments
    /// - `x_scaled`: A previously scaled data matrix.
    pub fn inverse_transform(&self, x_scaled: &Mat<f64>) -> Mat<f64> {
        let (nrows, ncols) = x_scaled.shape();

        Mat::from_fn(nrows, ncols, |i, j| {
            x_scaled[(i, j)] * self.x_sd[j] + self.x_mean[j]
        })
    }

    /// Returns the learned mean of each feature.
    pub fn mean(&self) -> &[f64] {
        &self.x_mean
    }

    /// Returns the learned standard deviation of each feature.
    pub fn std(&self) -> &[f64] {
        &self.x_sd
    }
}

#[cfg(test)]
mod tests {

    use super::StandardScaler;
    use faer::mat;

    #[test]
    fn test_fit_transform() {
        let x = mat![
            [0.1, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.5],
            [0.3, 0.2, 0.8],
            [0.6, 0.7, 0.5],
            [0.1, 0.3, 0.9],
        ];
        let scaler = StandardScaler::fit(&x);
        let x_scaled = scaler.transform(&x);
        let x_orig = scaler.inverse_transform(&x_scaled);
        let (n, p) = x.shape();
        for i in 0..n {
            for j in 0..p {
                assert!((x[(i, j)] - x_orig[(i, j)]).abs() < 1e-6);
            }
        }
    }
}
