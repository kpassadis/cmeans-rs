use faer::Mat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The shared intermediate struct Serde uses to read/write JSON
#[derive(Serialize, Deserialize)]
struct MatProxy {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

/// A proxy module to allow serializing and deserializing `faer::Mat<f64>`
/// Usage: Attach `#[serde(with = "crate::serialization::mat_serde")]`
pub mod mat_serde {
    use super::*;

    pub fn serialize<S>(mat: &Mat<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let mut data = Vec::with_capacity(nrows * ncols);

        for i in 0..nrows {
            for j in 0..ncols {
                data.push(mat[(i, j)]);
            }
        }

        let proxy = MatProxy { nrows, ncols, data };
        proxy.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Mat<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = MatProxy::deserialize(deserializer)?;

        if proxy.data.len() != proxy.nrows * proxy.ncols {
            return Err(serde::de::Error::custom(
                "Matrix dimensions do not match flattened data length",
            ));
        }

        Ok(Mat::from_fn(proxy.nrows, proxy.ncols, |i, j| {
            proxy.data[i * proxy.ncols + j]
        }))
    }
}

/// A proxy module to allow serializing and deserializing `Option<faer::Mat<f64>>`
/// Usage: Attach `#[serde(with = "crate::serialization::option_mat_serde")]`
pub mod option_mat_serde {
    use super::*;

    pub fn serialize<S>(mat_opt: &Option<Mat<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Map the Option<Mat> directly into an Option<MatProxy>
        let proxy_opt = mat_opt.as_ref().map(|mat| {
            let nrows = mat.nrows();
            let ncols = mat.ncols();
            let mut data = Vec::with_capacity(nrows * ncols);

            for i in 0..nrows {
                for j in 0..ncols {
                    data.push(mat[(i, j)]);
                }
            }
            MatProxy { nrows, ncols, data }
        });

        // Serde knows how to safely serialize an Option of a derivable struct!
        proxy_opt.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Mat<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Tell Serde to attempt extracting an Option<MatProxy>
        let proxy_opt: Option<MatProxy> = Option::deserialize(deserializer)?;

        match proxy_opt {
            Some(proxy) => {
                if proxy.data.len() != proxy.nrows * proxy.ncols {
                    return Err(serde::de::Error::custom(
                        "Matrix dimensions do not match flattened data length",
                    ));
                }

                Ok(Some(Mat::from_fn(proxy.nrows, proxy.ncols, |i, j| {
                    proxy.data[i * proxy.ncols + j]
                })))
            }
            None => Ok(None),
        }
    }
}
