//! Quality control functions for radar data
//!
//! This module provides QC functions that operate on numpy arrays
//! and return mask arrays indicating valid/invalid data.

mod rhohv;
mod sun_spike;

pub use rhohv::rhohv_threshold;
pub use sun_spike::sun_spike;

use pyo3::prelude::*;

/// Register the qc submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let qc_module = PyModule::new(parent_module.py(), "qc")?;

    qc_module.add_function(wrap_pyfunction!(rhohv::rhohv_threshold_py, &qc_module)?)?;
    qc_module.add_function(wrap_pyfunction!(sun_spike::sun_spike_py, &qc_module)?)?;

    parent_module.add_submodule(&qc_module)?;

    // Set the module path correctly for imports
    parent_module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("radrs.qc", qc_module)?;

    Ok(())
}
