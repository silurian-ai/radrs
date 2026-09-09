//! High-level iterators for NEXRAD data
//!
//! This module provides lazy iterators over archived volumes.

mod l2_archive;
mod peek;

pub use l2_archive::{
    NexradL2ArchiveInfo, NexradL2ArchiveIter, NexradL2ArchiveIterConfig, NexradL2ArchiveIterator,
};
pub use peek::{
    PEEK_FAST_INITIAL, PEEK_MAX_RECORDS, PEEK_SCAN_MAX, VOLUME_HEADER_SIZE, VolumeMeta,
    peek_volume, peek_volume_bytes, peek_volume_with_mode,
};

use pyo3::prelude::*;

/// Register iterator functions in the parent module
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<peek::VolumeMeta>()?;
    parent_module.add_class::<l2_archive::NexradL2ArchiveInfo>()?;
    parent_module.add_class::<l2_archive::NexradL2ArchiveIter>()?;
    parent_module.add_function(wrap_pyfunction!(l2_archive::list_nexrad_l2_archive_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(peek::peek_volume_py, parent_module)?)?;
    Ok(())
}
