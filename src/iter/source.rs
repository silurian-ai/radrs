//! Date-based volume listing for the Unidata NEXRAD Level 2 archive.

use crate::error::{RadrsError, Result};
use crate::fetch::{ARCHIVE_STORE, RUNTIME};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use futures::stream::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use pyo3::prelude::*;
use pyo3::types::PyDateTime;

/// Information about a volume from S3 listing
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct VolumeInfo {
    /// Volume filename (e.g., "KTLX20240702_000556_V06")
    #[pyo3(get)]
    pub name: String,
    /// File size in bytes
    #[pyo3(get)]
    pub size: u64,
    /// Last modified timestamp
    last_modified: DateTime<Utc>,
}

#[pymethods]
impl VolumeInfo {
    fn __repr__(&self) -> String {
        format!(
            "VolumeInfo(name='{}', size={}, last_modified='{}')",
            self.name,
            self.size,
            self.last_modified.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    fn __str__(&self) -> String {
        self.name.clone()
    }

    /// Get last_modified as Python datetime
    #[getter]
    fn last_modified<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDateTime>> {
        let timestamp = self.last_modified.timestamp();
        let microseconds = self.last_modified.timestamp_subsec_micros();
        PyDateTime::from_timestamp(
            py,
            timestamp as f64 + microseconds as f64 / 1_000_000.0,
            None,
        )
    }

    /// Get S3 URL for this volume
    fn s3_url(&self, site: &str, date: &str) -> PyResult<String> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|e| RadrsError::Parse(format!("Invalid date format: {}", e)))?;
        Ok(format!(
            "s3://unidata-nexrad-level2/{}/{:02}/{:02}/{}/{}",
            date.format("%Y"),
            date.month(),
            date.day(),
            site,
            self.name
        ))
    }
}

/// List available volumes with full metadata (size, last_modified) for a single date.
pub async fn list_volumes_with_info(site: &str, date: NaiveDate) -> Result<Vec<VolumeInfo>> {
    let prefix = format!(
        "{}/{:02}/{:02}/{}/",
        date.format("%Y"),
        date.month(),
        date.day(),
        site
    );

    let prefix_path = ObjectPath::from(prefix);

    let mut volumes = Vec::new();
    let mut list_stream = ARCHIVE_STORE.list(Some(&prefix_path));

    while let Some(result) = list_stream.next().await {
        match result {
            Ok(meta) => {
                let path = meta.location.to_string();
                if let Some(filename) = path.rsplit('/').next() {
                    // Filter to actual volume files:
                    // - Must start with site name
                    // - Must contain version indicator (_V0 or _V1)
                    // - Exclude metadata-only files (_MDM suffix)
                    if filename.starts_with(site)
                        && (filename.contains("_V0") || filename.contains("_V1"))
                        && !filename.ends_with("_MDM")
                    {
                        volumes.push(VolumeInfo {
                            name: filename.to_string(),
                            size: meta.size as u64,
                            last_modified: meta.last_modified,
                        });
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Error listing object: {}", e);
            }
        }
    }

    // Sort by name (which includes timestamp, so this is chronological)
    volumes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(volumes)
}

/// List available volumes (Python wrapper) - returns VolumeInfo with size and last_modified
#[pyfunction]
#[pyo3(name = "list_volumes")]
pub fn list_volumes_py(_py: Python<'_>, site: &str, date: &str) -> PyResult<Vec<VolumeInfo>> {
    let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| RadrsError::Parse(format!("Invalid date format: {}", e)))?;

    let volumes = RUNTIME.block_on(list_volumes_with_info(site, date))?;

    Ok(volumes)
}
