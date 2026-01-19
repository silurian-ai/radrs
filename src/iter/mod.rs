//! High-level iterators for NEXRAD data
//!
//! This module provides lazy iterators over S3 volumes and streaming.

mod realtime;
mod volumes;

pub use realtime::{stream_archive, stream_realtime_py, stream_archive_py};
pub use volumes::{iter_volumes, list_volumes};

use pyo3::prelude::*;

/// Register iterator functions in the parent module
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_function(wrap_pyfunction!(volumes::list_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(volumes::iter_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(volumes::iter_volumes_async_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(realtime::stream_archive_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(realtime::stream_realtime_py, parent_module)?)?;
    Ok(())
}
