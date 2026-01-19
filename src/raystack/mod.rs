//! Raystack format tools for ML training
//!
//! This module provides tools for converting NEXRAD data to the raystack format,
//! which is optimized for machine learning training pipelines.

mod convert;
mod fold;
mod parse;
#[cfg(feature = "zarr")]
mod zarr;

pub use convert::{from_datatree_py, to_datatree_py};
pub use fold::{fold_ranges, fold_ranges_into};
pub use parse::{parse_optimized, parse_py, RaystackData, SweepInfo};
#[cfg(feature = "zarr")]
pub use zarr::{open_zarr, to_zarr};

use pyo3::prelude::*;

/// Register the raystack submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let raystack_module = PyModule::new(parent_module.py(), "raystack")?;

    raystack_module.add_function(wrap_pyfunction!(parse::parse_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(convert::from_datatree_py, &raystack_module)?)?;
    raystack_module.add_function(wrap_pyfunction!(convert::to_datatree_py, &raystack_module)?)?;

    #[cfg(feature = "zarr")]
    {
        raystack_module.add_function(wrap_pyfunction!(zarr::to_zarr_py, &raystack_module)?)?;
    }

    parent_module.add_submodule(&raystack_module)?;

    // Set the module path correctly for imports
    parent_module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("radrs.raystack", raystack_module)?;

    Ok(())
}
