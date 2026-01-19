//! xradar-compatible interface for NEXRAD data
//!
//! This module provides an interface that matches xradar's API for easy comparison
//! and drop-in replacement.

pub(crate) mod datatree;

pub use datatree::open_datatree;

use pyo3::prelude::*;

/// Register the xradar submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let xradar_module = PyModule::new(parent_module.py(), "xradar")?;
    xradar_module.add_function(wrap_pyfunction!(datatree::open_datatree_py, &xradar_module)?)?;
    xradar_module.add_function(wrap_pyfunction!(datatree::open_datatree_async_py, &xradar_module)?)?;
    parent_module.add_submodule(&xradar_module)?;

    // Set the module path correctly for imports
    parent_module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("radrs.xradar", xradar_module)?;

    Ok(())
}
