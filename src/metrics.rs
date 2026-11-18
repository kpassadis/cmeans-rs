use faer::{Mat, MatRef, RowRef};
use std::collections::HashMap;

/// If x is large, swapping the loop order so j is the outer loop over columns and i is the
/// inner loop over rows aligns with faer's column-major memory layout and enables LLVM auto-vectorization
/// # Returns
/// A column matrix of shape `[n x 1]` containing the distances.
pub(crate) fn weighted_sq_distances(
    v: RowRef<'_, f64>,
    x: MatRef<'_, f64>,
    g: RowRef<'_, f64>,
) -> Mat<f64> {
    let p = v.ncols();
    assert_eq!(x.ncols(), p);
    assert_eq!(g.ncols(), p);

    let n = x.nrows();
    Mat::from_fn(n, 1, |i, _| {
        let mut sum = 0.0;
        for j in 0..p {
            let diff = x[(i, j)] - v[j];
            sum += diff * g[j] * diff;
        }
        sum
    })
}

pub(crate) fn sq_euclidean_distances(v: RowRef<'_, f64>, x: MatRef<'_, f64>) -> Mat<f64> {
    let p = v.ncols();
    assert_eq!(x.ncols(), p);

    let n = x.nrows();
    Mat::from_fn(n, 1, |i, _| {
        let mut sum = 0.0;
        for j in 0..p {
            let diff = x[(i, j)] - v[j];
            sum += diff * diff;
        }
        sum
    })
}

pub trait ClusterMetrics {
    fn partition_coefficient(u: &Mat<f64>) -> f64;
    fn entropy_coefficient(u: &Mat<f64>) -> f64;
    fn evaluate(&self, x: &Mat<f64>) -> HashMap<String, f64>;
}
