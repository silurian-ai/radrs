//! radrs - Rust-Native NEXRAD Processing Pipeline
//!
//! A modular Rust-native NEXRAD processing pipeline with:
//! - `radrs.xradar` - xradar-compatible interface for easy comparison
//! - `radrs.raystack` - raystack format tools for ML training
//! - `radrs.qc` - Quality control functions
//! - High-level iterators built on top of these primitives
//!
//! # Example
//!
//! ```python
//! import radrs
//! import radrs.xradar as rxr
//! import radrs.raystack as rrs
//! import radrs.qc as qc
//!
//! # Load a NEXRAD file as xarray DataTree
//! dt = rxr.open_datatree("path/to/file.ar2v")
//!
//! # Or parse directly to raystack format for ML
//! rs = rrs.parse(file_bytes, fold_size=128)
//!
//! # Iterate over S3 archive
//! for dt in radrs.iter_volumes("KTLX", start="2024-03-15"):
//!     process(dt)
//! ```

pub mod cache;
pub mod error;
pub mod fetch;
pub mod iter;
pub mod parse;
pub mod qc;
pub mod raystack;
pub mod xradar;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn radrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    xradar::register_module(m)?;
    raystack::register_module(m)?;
    qc::register_module(m)?;

    // Register top-level iterator functions
    iter::register_functions(m)?;

    // Register cache functions (if feature enabled)
    cache::register_functions(m)?;

    Ok(())
}
