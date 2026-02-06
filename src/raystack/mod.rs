//! Raystack format tools for ML training
//!
//! This module provides tools for converting NEXRAD data to the raystack format,
//! which is optimized for machine learning training pipelines.

mod batch;
mod convert;
mod parse;

pub use batch::{BatchedRaystackPy, parse_single_volume};
pub use convert::{from_xradar_datatree_py, to_xradar_datatree_py};
pub(crate) use parse::parse_qc_ops;
pub use parse::{QcOp, parse_py};

use pyo3::prelude::*;

/// Register the raystack submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let raystack_module = PyModule::new(parent_module.py(), "raystack")?;

    raystack_module.add_function(wrap_pyfunction!(parse::parse_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(
        parse::open_raystack_datatree_py,
        &raystack_module
    )?)?;
    raystack_module.add_function(wrap_pyfunction!(
        parse::open_raystack_datatree_async_py,
        &raystack_module
    )?)?;
    raystack_module.add_function(wrap_pyfunction!(
        convert::from_xradar_datatree_py,
        &raystack_module
    )?)?;
    raystack_module.add_function(wrap_pyfunction!(
        convert::to_xradar_datatree_py,
        &raystack_module
    )?)?;
    raystack_module.add_function(wrap_pyfunction!(
        convert::to_raystack_datatree_py,
        &raystack_module
    )?)?;

    raystack_module.add_class::<batch::BatchedRaystackPy>()?;

    parent_module.add_submodule(&raystack_module)?;
    Ok(())
}
