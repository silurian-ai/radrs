//! Realtime streaming of NEXRAD data.
//!
//! Streaming from the Unidata chunks bucket requires chunk accumulation and is not yet
//! implemented.

use pyo3::prelude::*;

/// Stream realtime NEXRAD data (not yet implemented)
///
/// This would stream from the Unidata chunks bucket for near-realtime data.
/// Requires accumulating chunks into complete volumes before parsing.
#[pyfunction]
#[pyo3(name = "stream_realtime")]
pub fn stream_realtime_py(_py: Python<'_>, _site: &str) -> PyResult<Py<PyAny>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Realtime chunk streaming is not yet implemented.",
    ))
}
