//! Raystack format tools for ML training
//!
//! This module provides tools for converting NEXRAD data to the raystack format,
//! which is optimized for machine learning training pipelines.

mod convert;
mod fold;
mod parse;

pub use convert::{from_xradar_datatree_py, to_xradar_datatree_py};
pub use fold::{fold_ranges, fold_ranges_into};
pub use parse::{parse_optimized, parse_py, QcOp, RaystackData, SweepInfo};
pub(crate) use parse::{parse_qc_ops, raystack_to_python};

use pyo3::prelude::*;

/// Register the raystack submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let raystack_module = PyModule::new(parent_module.py(), "raystack")?;

    raystack_module.add_function(wrap_pyfunction!(parse::parse_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(parse::open_raystack_datatree_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(parse::open_raystack_datatree_async_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(convert::from_xradar_datatree_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(convert::to_xradar_datatree_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(convert::to_raystack_datatree_py, &raystack_module)?)?;

    parent_module.add_submodule(&raystack_module)?;
    Ok(())
}
