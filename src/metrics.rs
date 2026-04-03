use faer::Mat;
use std::collections::HashMap;

pub trait ClusterMetrics {
    fn partition_coefficient(u: &Mat<f64>) -> f64;
    fn entropy_coefficient(u: &Mat<f64>) -> f64;
    fn evaluate(&self, x: &Mat<f64>) -> HashMap<String, f64>;
}
