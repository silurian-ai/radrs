//! High-level iterators for NEXRAD data
//!
//! This module provides lazy iterators over S3 volumes and streaming.

mod l2_archive;
mod peek;
mod realtime;

pub use l2_archive::{NexradL2ArchiveInfo, NexradL2ArchiveIter, NexradL2ArchiveIterator, list_nexrad_l2_archive_volumes_py};
pub use peek::{VolumeMeta, peek_volume_bytes, VOLUME_HEADER_SIZE, PEEK_FAST_INITIAL, PEEK_SCAN_MAX, PEEK_MAX_RECORDS};
pub use realtime::{stream_archive, stream_archive_py, stream_realtime_py};

use pyo3::prelude::*;

/// Register iterator functions in the parent module
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<peek::VolumeMeta>()?;
    parent_module.add_class::<l2_archive::NexradL2ArchiveInfo>()?;
    parent_module.add_class::<l2_archive::NexradL2ArchiveIter>()?;
    parent_module.add_function(wrap_pyfunction!(l2_archive::list_nexrad_l2_archive_volumes_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(peek::peek_volume_py, parent_module)?)?;
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
