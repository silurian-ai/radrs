//! radrs - Rust-Native NEXRAD Processing Pipeline
//!
//! A modular Rust-native NEXRAD processing pipeline with:
//! - `radrs.xradar` - xradar-compatible interface for easy comparison
//! - `radrs.raystack` - raystack format tools for ML training
//! - `radrs.qc` - Quality control functions
//! - High-level iterators built on top of these primitives
//!
//! # Example
//!
//! ```python
//! import radrs
//! import radrs.xradar as rxr
//! import radrs.raystack as rrs
//! import radrs.qc as qc
//!
//! # Load a NEXRAD file as xarray DataTree
//! dt = rxr.open_datatree("path/to/file.ar2v")
//!
//! # Or parse directly to raystack format for ML
//! rs = rrs.parse(file_bytes, fold_size=128)
//!
//! # Iterate over S3 archive
//! from datetime import datetime
//! archive = radrs.NexradL2ArchiveIter(
//!     base_uri="s3://unidata-nexrad-level2",
//!     start_time=datetime(2024, 3, 15),
//!     end_time=datetime(2024, 3, 16),
//!     storage_options={"anon": "true"},
//!     site_filter=["KTLX"],
//! )
//! for info in archive:
//!     dt = rxr.open_datatree(info.uri)
//!     process(dt)
//! ```

// Use mimalloc for better multi-threaded allocation performance on Linux
#[cfg(all(feature = "fast-alloc", target_os = "linux"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod error;
pub mod fetch;
pub mod iter;
pub mod constants;
pub mod metadata;
pub mod metadata_build;
pub mod ops;
pub mod qc;
pub mod raystack;
pub mod xradar;

use pyo3::prelude::*;
use pyo3::types::PyMapping;
use std::env;
use std::sync::Once;

static TRACING_INIT: Once = Once::new();

/// Read RADRS_LOG env var from Python's os.environ (handles venv correctly)
fn log_filter_from_env(py: Python<'_>) -> PyResult<Option<String>> {
    let os = py.import("os")?;
    let environ = os.getattr("environ")?;
    let environ: &Bound<'_, PyMapping> = environ.cast()?;
    let value = environ
        .get_item("RADRS_LOG")
        .ok()
        .and_then(|v| v.extract().ok());
    Ok(value)
}

/// Initialize tracing with optional filter directive (e.g., "radrs=debug,radrs::iter=trace")
fn initialize_tracing(filter: Option<&str>) {
    use tracing_subscriber::{EnvFilter, fmt};

    TRACING_INIT.call_once(|| {
        let filter = filter
            .map(|s| s.to_string())
            .unwrap_or_else(|| "radrs=warn".to_string());

        let subscriber = fmt::Subscriber::builder()
            .with_env_filter(EnvFilter::new(filter))
            .with_target(true)
            .with_ansi(false) // No ANSI codes for cleaner output
            .finish();

        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

/// Initialize logging (uses RADRS_LOG env var if set, otherwise defaults to warn level).
///
/// Call this explicitly to enable logging output from radrs.
/// By default, no logging is performed for zero overhead.
///
/// Example:
///     import radrs
///     radrs.initialize_logs()  # Uses RADRS_LOG env var or defaults to warn
///
/// Or set env var before running:
///     RADRS_LOG=radrs=debug python script.py
#[pyfunction]
fn initialize_logs(py: Python<'_>) -> PyResult<()> {
    if env::var("RADRS_NO_LOGS").is_err() {
        let filter = log_filter_from_env(py)?;
        initialize_tracing(filter.as_deref());
    }
    Ok(())
}

/// Set log filter directive explicitly.
///
/// Example:
///     import radrs
///     radrs.set_log_filter("radrs=debug")  # Enable debug logging
///     radrs.set_log_filter("radrs::iter=trace")  # Trace just the iterator module
#[pyfunction]
#[pyo3(signature = (filter=None))]
fn set_log_filter(py: Python<'_>, filter: Option<String>) -> PyResult<()> {
    let filter = filter.or_else(|| log_filter_from_env(py).ok().flatten());
    initialize_tracing(filter.as_deref());
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _radrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register logging functions (opt-in, no overhead by default)
    m.add_function(wrap_pyfunction!(initialize_logs, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_filter, m)?)?;

    // Register submodules
    xradar::register_module(m)?;
    raystack::register_module(m)?;
    qc::register_module(m)?;
    ops::register_module(m)?;

    // Register top-level iterator functions
    iter::register_functions(m)?;

    Ok(())
}
