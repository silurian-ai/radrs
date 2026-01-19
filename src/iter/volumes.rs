//! Volume iteration over S3 archive

use crate::error::{RadrsError, Result};
use chrono::{Datelike, NaiveDate};
use futures::stream::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

const NEXRAD_BUCKET: &str = "noaa-nexrad-level2";

/// List available volumes for a site and date
pub async fn list_volumes(site: &str, date: NaiveDate) -> Result<Vec<String>> {
    let store = AmazonS3Builder::new()
        .with_bucket_name(NEXRAD_BUCKET)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .build()?;

    let prefix = format!(
        "{}/{:02}/{:02}/{}/",
        date.format("%Y"),
        date.month(),
        date.day(),
        site
    );

    let prefix_path = ObjectPath::from(prefix);

    let mut volumes = Vec::new();
    let mut list_stream = store.list(Some(&prefix_path));

    while let Some(result) = list_stream.next().await {
        match result {
            Ok(meta) => {
                let path = meta.location.to_string();
                // Extract just the filename
                if let Some(filename) = path.rsplit('/').next() {
                    // Filter out MDM files and other non-volume files
                    if filename.starts_with(site)
                        && (filename.contains("_V0") || filename.contains("_V1"))
                    {
                        volumes.push(filename.to_string());
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Error listing object: {}", e);
            }
        }
    }

    volumes.sort();
    Ok(volumes)
}

/// List available volumes (Python wrapper)
#[pyfunction]
#[pyo3(name = "list_volumes")]
pub fn list_volumes_py(_py: Python<'_>, site: &str, date: &str) -> PyResult<Vec<String>> {
    let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| RadrsError::Parse(format!("Invalid date format: {}", e)))?;

    let rt = tokio::runtime::Runtime::new()?;
    let volumes = rt.block_on(list_volumes(site, date))?;

    Ok(volumes)
}

/// Iterate over volumes for a site and date range
#[pyfunction]
#[pyo3(name = "iter_volumes", signature = (site, start, end = None, schema = None, qc = None, fold_size = None))]
pub fn iter_volumes_py(
    py: Python<'_>,
    site: &str,
    start: &str,
    end: Option<&str>,
    schema: Option<&str>,
    qc: Option<bool>,
    fold_size: Option<usize>,
) -> PyResult<PyObject> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .map_err(|e| RadrsError::Parse(format!("Invalid start date: {}", e)))?;
    let end_date = end
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d"))
        .transpose()
        .map_err(|e| RadrsError::Parse(format!("Invalid end date: {}", e)))?
        .unwrap_or(start_date);

    let schema = schema.unwrap_or("xradar").to_string();
    let fold_size = fold_size.unwrap_or(128);
    let qc_enabled = qc.unwrap_or(false);

    // Create an iterator class in Python
    let iterator_code = r#"
from datetime import timedelta

class VolumeIterator:
    def __init__(self, site, start_date, end_date, schema, fold_size, qc_enabled):
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        self.schema = schema
        self.fold_size = fold_size
        self.qc_enabled = qc_enabled
        self._volumes = None
        self._current_date = start_date
        self._current_index = 0
        self._radrs = __import__('radrs')

    def __iter__(self):
        return self

    def __next__(self):
        import urllib.request

        while self._current_date <= self.end_date:
            # Get volumes for current date if not loaded
            if self._volumes is None:
                self._volumes = self._radrs.list_volumes(
                    self.site,
                    self._current_date.strftime('%Y-%m-%d')
                )
                self._current_index = 0

            # Check if we have more volumes for current date
            if self._current_index < len(self._volumes):
                volume = self._volumes[self._current_index]
                self._current_index += 1

                # Build HTTPS URL (public bucket)
                https_url = f"https://noaa-nexrad-level2.s3.amazonaws.com/{self._current_date.year}/{self._current_date.month:02d}/{self._current_date.day:02d}/{self.site}/{volume}"

                try:
                    with urllib.request.urlopen(https_url) as response:
                        data = response.read()

                    if self.schema == "raystack":
                        return self._radrs.raystack.parse(
                            data,
                            fold_size=self.fold_size
                        )
                    else:
                        return self._radrs.xradar.open_datatree(data)
                except Exception as e:
                    # Skip failed volumes
                    continue

            # Move to next date
            self._current_date += timedelta(days=1)
            self._volumes = None

        raise StopIteration

VolumeIterator
"#;

    let globals = PyDict::new(py);
    let code = CString::new(iterator_code).unwrap();
    py.run(code.as_c_str(), Some(&globals), None)?;

    let iterator_class = globals.get_item("VolumeIterator")?.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to create VolumeIterator class")
    })?;

    let datetime = py.import("datetime")?;
    let start_dt = datetime
        .getattr("date")?
        .call1((start_date.year(), start_date.month(), start_date.day()))?;
    let end_dt = datetime
        .getattr("date")?
        .call1((end_date.year(), end_date.month(), end_date.day()))?;

    let iterator = iterator_class.call1((site, start_dt, end_dt, &schema, fold_size, qc_enabled))?;

    Ok(iterator.into())
}

/// Iterate over volumes (internal async version)
pub async fn iter_volumes(
    _site: &str,
    _start: NaiveDate,
    _end: NaiveDate,
) -> Result<()> {
    // Internal implementation would go here
    unimplemented!("Use iter_volumes_py for Python interface")
}

