//! Volume sources for iterators

use crate::error::{RadrsError, Result};
use crate::fetch::{ARCHIVE_STORE, RUNTIME, fetch_archive_file};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use futures::stream::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use pyo3::prelude::*;
use pyo3::types::PyDateTime;

/// Information about a volume from S3 listing
#[pyclass]
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

#[derive(Clone)]
pub(crate) enum VolumeRef {
    NexradArchive {
        site: String,
        date: NaiveDate,
        volume: String,
    },
}

impl VolumeRef {
    pub async fn fetch(&self) -> Result<Vec<u8>> {
        match self {
            VolumeRef::NexradArchive { site, date, volume } => {
                fetch_archive_file(site, date.year(), date.month(), date.day(), volume).await
            }
        }
    }
}


#[derive(Clone)]
pub(crate) struct NexradArchiveConfig {
    site: String,
    start: NaiveDate,
    end: NaiveDate,
}

impl NexradArchiveConfig {
    fn to_state(&self) -> NexradArchiveState {
        NexradArchiveState {
            site: self.site.clone(),
            current_date: self.start,
            end_date: self.end,
            volumes: Vec::new(),
            index: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) enum VolumeSourceConfig {
    NexradArchive(NexradArchiveConfig),
}

impl VolumeSourceConfig {
    fn to_state(&self) -> VolumeSourceState {
        match self {
            VolumeSourceConfig::NexradArchive(config) => {
                VolumeSourceState::NexradArchive(config.to_state())
            }
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct VolumeSource {
    config: VolumeSourceConfig,
}

#[pymethods]
impl VolumeSource {
    #[staticmethod]
    #[pyo3(signature = (site, start, end=None))]
    fn nexrad(site: &str, start: &str, end: Option<&str>) -> PyResult<Self> {
        let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
            .map_err(|e| RadrsError::Parse(format!("Invalid start date: {}", e)))?;
        let end_date = end
            .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d"))
            .transpose()
            .map_err(|e| RadrsError::Parse(format!("Invalid end date: {}", e)))?
            .unwrap_or(start_date);

        if end_date < start_date {
            return Err(
                RadrsError::Parse("End date cannot be before start date".to_string()).into(),
            );
        }

        Ok(VolumeSource {
            config: VolumeSourceConfig::NexradArchive(NexradArchiveConfig {
                site: site.to_string(),
                start: start_date,
                end: end_date,
            }),
        })
    }
}

impl VolumeSource {
    pub(crate) fn to_state(&self) -> VolumeSourceState {
        self.config.to_state()
    }
}

pub(crate) enum VolumeSourceState {
    NexradArchive(NexradArchiveState),
}

impl VolumeSourceState {
    pub async fn next_ref(&mut self) -> Result<Option<VolumeRef>> {
        match self {
            VolumeSourceState::NexradArchive(state) => state.next_ref().await,
        }
    }
}

pub(crate) struct NexradArchiveState {
    site: String,
    current_date: NaiveDate,
    end_date: NaiveDate,
    volumes: Vec<String>,
    index: usize,
}

impl NexradArchiveState {
    async fn next_ref(&mut self) -> Result<Option<VolumeRef>> {
        loop {
            if self.current_date > self.end_date {
                return Ok(None);
            }

            if !self.volumes.is_empty() && self.index >= self.volumes.len() {
                self.current_date = self.current_date.succ_opt().unwrap_or(self.current_date);
                self.volumes.clear();
                self.index = 0;
                continue;
            }

            if self.volumes.is_empty() {
                self.volumes = list_volumes(&self.site, self.current_date).await?;
                self.index = 0;

                if self.volumes.is_empty() {
                    self.current_date = self.current_date.succ_opt().unwrap_or(self.current_date);
                    continue;
                }
            }

            if self.index < self.volumes.len() {
                let volume = self.volumes[self.index].clone();
                self.index += 1;
                return Ok(Some(VolumeRef::NexradArchive {
                    site: self.site.clone(),
                    date: self.current_date,
                    volume,
                }));
            }
        }
    }
}

/// List available volumes for a site and date (internal, returns just names for iteration)
pub async fn list_volumes(site: &str, date: NaiveDate) -> Result<Vec<String>> {
    let volumes = list_volumes_with_info(site, date).await?;
    Ok(volumes.into_iter().map(|v| v.name).collect())
}

/// List available volumes with full metadata (size, last_modified)
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
                    if filename.starts_with(site)
                        && (filename.contains("_V0") || filename.contains("_V1"))
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
