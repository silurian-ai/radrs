//! Optional caching for NEXRAD data

#[cfg(feature = "cache")]
mod icechunk;

#[cfg(feature = "cache")]
pub use icechunk::open_cache;

use pyo3::prelude::*;

/// Register cache functions (when cache feature is enabled)
#[cfg(feature = "cache")]
pub fn register_functions(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_function(wrap_pyfunction!(icechunk::open_cache_py, parent_module)?)?;
    Ok(())
}

#[cfg(not(feature = "cache"))]
pub fn register_functions(_parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // No-op when cache feature is disabled
    Ok(())
}
