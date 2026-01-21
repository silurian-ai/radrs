//! High-level iterators for NEXRAD data
//!
//! This module provides lazy iterators over S3 volumes and streaming.

mod realtime;
mod source;
mod volumes;

pub use realtime::{stream_archive, stream_realtime_py, stream_archive_py};
pub use source::{list_volumes, VolumeSource};

use pyo3::prelude::*;

/// Register iterator functions in the parent module
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<source::VolumeSource>()?;
    parent_module.add_function(wrap_pyfunction!(source::list_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(volumes::iter_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(volumes::iter_volumes_async_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(realtime::stream_archive_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(realtime::stream_realtime_py, parent_module)?)?;
    Ok(())
}
