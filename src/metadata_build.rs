//! Shared builders for metadata datasets and variables.

use crate::constants::{
    CF_CONVENTIONS, DEFAULT_FOLLOW_MODE, DEFAULT_PRT_MODE, DEFAULT_SWEEP_MODE, INSTRUMENT_TYPE,
    PLATFORM_TYPE,
};
use crate::metadata::ScanMeta;
use chrono::{TimeZone, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build root attributes for radar datatrees (raystack + xradar).
pub fn build_root_attrs<'py>(
    py: Python<'py>,
    pattern_number: u16,
    instrument_name: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let root_attrs = PyDict::new(py);
    root_attrs.set_item("Conventions", CF_CONVENTIONS)?;
    root_attrs.set_item("instrument_type", INSTRUMENT_TYPE)?;
    root_attrs.set_item("platform_type", PLATFORM_TYPE)?;
    root_attrs.set_item("volume_coverage_pattern", pattern_number)?;
    root_attrs.set_item("scan_name", format!("VCP-{}", pattern_number))?;
    if let Some(name) = instrument_name {
        root_attrs.set_item("instrument_name", name)?;
    }
    Ok(root_attrs)
}

/// Format a millisecond timestamp as ISO 8601 (UTC).
pub fn format_timestamp_iso(timestamp_ms: i64) -> String {
    let dt = Utc.timestamp_millis_opt(timestamp_ms).unwrap();
    dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Build root data variables for xradar DataTrees.
pub fn build_root_vars<'py>(
    py: Python<'py>,
    meta: &ScanMeta,
    time_start: Option<i64>,
    time_end: Option<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let root_vars = PyDict::new(py);

    if let Some(volume_number) = meta.volume_number {
        root_vars.set_item("volume_number", volume_number)?;
    }
    // NEXRAD is a fixed platform; platform_number is not provided in the file format.
    root_vars.set_item("platform_number", 0)?;
    root_vars.set_item("platform_type", PLATFORM_TYPE)?;
    root_vars.set_item("instrument_type", INSTRUMENT_TYPE)?;

    if let Some(lat) = meta.latitude {
        root_vars.set_item("latitude", lat)?;
    }
    if let Some(lon) = meta.longitude {
        root_vars.set_item("longitude", lon)?;
    }
    if let Some(alt) = meta.altitude {
        root_vars.set_item("altitude", alt)?;
    }

    if let Some(ts) = time_start {
        root_vars.set_item("time_coverage_start", format_timestamp_iso(ts))?;
    }
    if let Some(ts) = time_end {
        root_vars.set_item("time_coverage_end", format_timestamp_iso(ts))?;
    }

    Ok(root_vars)
}

/// Add sweep metadata variables for sweep datasets (raystack).
pub fn set_sweep_mode_vars<'py>(
    vars: &Bound<'py, PyDict>,
    n_sweeps: usize,
) -> PyResult<()> {
    vars.set_item(
        "sweep_mode",
        (("sweep_time",), vec![DEFAULT_SWEEP_MODE; n_sweeps]),
    )?;
    vars.set_item(
        "prt_mode",
        (("sweep_time",), vec![DEFAULT_PRT_MODE; n_sweeps]),
    )?;
    vars.set_item(
        "follow_mode",
        (("sweep_time",), vec![DEFAULT_FOLLOW_MODE; n_sweeps]),
    )?;
    Ok(())
}

/// Add sweep metadata variables as scalars for per-sweep datasets (xradar).
pub fn set_sweep_mode_scalars<'py>(vars: &Bound<'py, PyDict>) -> PyResult<()> {
    vars.set_item("sweep_mode", DEFAULT_SWEEP_MODE)?;
    vars.set_item("prt_mode", DEFAULT_PRT_MODE)?;
    vars.set_item("follow_mode", DEFAULT_FOLLOW_MODE)?;
    Ok(())
}
