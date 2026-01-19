//! Error types for radrs

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

/// Result type for radrs operations
pub type Result<T> = std::result::Result<T, RadrsError>;

/// Error types for radrs
#[derive(Error, Debug)]
pub enum RadrsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NEXRAD data error: {0}")]
    NexradData(#[from] nexrad_data::result::Error),

    #[error("NEXRAD decode error: {0}")]
    NexradDecode(#[from] nexrad_decode::result::Error),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Missing data: {0}")]
    MissingData(String),

    #[error("Python error: {0}")]
    Python(String),

    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    #[error("Unsupported file format")]
    UnsupportedFormat,

    #[error("Parse error: {0}")]
    Parse(String),
}

impl From<RadrsError> for PyErr {
    fn from(err: RadrsError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
