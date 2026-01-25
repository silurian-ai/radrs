//! High-level iterators for NEXRAD data
//!
//! This module provides lazy iterators over S3 volumes and streaming.

mod meta;
mod peek;
mod realtime;
mod source;
mod volumes;

pub use peek::VolumeMeta;
pub use realtime::{stream_archive, stream_archive_py, stream_realtime_py};
pub use source::{VolumeInfo, VolumeSource, list_volumes};

use pyo3::prelude::*;

/// Register iterator functions in the parent module
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<source::VolumeSource>()?;
    parent_module.add_class::<source::VolumeInfo>()?;
    parent_module.add_class::<peek::VolumeMeta>()?;
    parent_module.add_function(wrap_pyfunction!(source::list_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(peek::peek_volume_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(meta::iter_meta_urls_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(meta::iter_meta_urls_async_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(
        meta::iter_meta_candidates_py,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        meta::iter_meta_candidates_async_py,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(volumes::iter_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(
        volumes::iter_volumes_async_py,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        realtime::stream_archive_py,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        realtime::stream_realtime_py,
        parent_module
    )?)?;
    Ok(())
}
