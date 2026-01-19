//! Volume iteration over S3 archive

use crate::error::{RadrsError, Result};
use crate::fetch::{fetch_archive_file, ARCHIVE_STORE, RUNTIME};
use crate::raystack;
use crate::xradar;
use chrono::{Datelike, NaiveDate};
use futures::stream::StreamExt;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// List available volumes for a site and date
pub async fn list_volumes(site: &str, date: NaiveDate) -> Result<Vec<String>> {
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

    let volumes = RUNTIME.block_on(list_volumes(site, date))?;

    Ok(volumes)
}

/// Sync iterator with prefetch support.
/// Uses the same VolumeIterState as the async iterator, driven via block_on.
#[pyclass]
struct VolumeIterator {
    state: Arc<std::sync::Mutex<VolumeIterState>>,
    schema: String,
    fold_size: usize,
    _qc_enabled: bool,
}

#[pymethods]
impl VolumeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Get next bytes from prefetch pipeline (blocking)
        let data: Option<Vec<u8>> = py.detach(|| {
            let state = self.state.clone();
            RUNTIME.block_on(async move {
                let mut guard = state.lock().expect("state lock poisoned");
                guard.next_bytes().await
            })
        })?;

        let Some(data) = data else {
            return Err(pyo3::exceptions::PyStopIteration::new_err(()));
        };

        if self.schema == "raystack" {
            return raystack::parse_py(py, &data, Some(self.fold_size), None);
        }

        let scan = py.detach(|| xradar::open_datatree(data))?;
        xradar::datatree::scan_to_datatree(py, &scan)
    }
}

struct VolumeIterState {
    site: String,
    current_date: NaiveDate,
    end_date: NaiveDate,
    volumes: Vec<String>,
    index: usize,
    prefetch: usize,
    in_flight: VecDeque<JoinHandle<Result<Vec<u8>>>>,
}

impl VolumeIterState {
    async fn next_volume_id(&mut self) -> Result<Option<(NaiveDate, String)>> {
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
                return Ok(Some((self.current_date, volume)));
            }
        }
    }

    async fn fill_prefetch(&mut self) -> Result<()> {
        while self.in_flight.len() < self.prefetch {
            let Some((date, volume)) = self.next_volume_id().await? else {
                break;
            };

            let site = self.site.clone();
            let handle = tokio::spawn(async move {
                fetch_archive_file(&site, date.year(), date.month(), date.day(), &volume).await
            });
            self.in_flight.push_back(handle);
        }

        Ok(())
    }

    async fn next_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        if self.in_flight.is_empty() {
            self.fill_prefetch().await?;
        }

        while let Some(handle) = self.in_flight.pop_front() {
            let result = handle
                .await
                .map_err(|e| RadrsError::Python(format!("Fetch task failed: {}", e)))?;
            match result {
                Ok(bytes) => {
                    self.fill_prefetch().await?;
                    return Ok(Some(bytes));
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch volume: {}", e);
                    self.fill_prefetch().await?;
                    continue;
                }
            }
        }

        Ok(None)
    }
}

#[pyclass]
struct VolumeIteratorAsync {
    state: Arc<Mutex<VolumeIterState>>,
    schema: String,
    fold_size: usize,
    _qc_enabled: bool,
}

#[pymethods]
impl VolumeIteratorAsync {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = slf.state.clone();
        let schema = slf.schema.clone();
        let fold_size = slf.fold_size;
        let _qc_enabled = slf._qc_enabled;

        let awaitable = future_into_py(py, async move {
            let bytes = {
                let mut guard = state.lock().await;
                guard.next_bytes().await?
            };

            let Some(data) = bytes else {
                return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(()));
            };

            if schema == "raystack" {
                let raystack = tokio::task::spawn_blocking(move || {
                    raystack::parse_optimized(&data, fold_size)
                })
                .await
                .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;

                return Python::attach(|py| raystack::raystack_to_python(py, raystack, None))
                    .map_err(Into::into);
            }

            let scan = tokio::task::spawn_blocking(move || xradar::open_datatree(data))
                .await
                .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;

            Python::attach(|py| xradar::datatree::scan_to_datatree(py, &scan)).map_err(Into::into)
        })?;

        Ok(awaitable.into())
    }
}

/// Iterate over volumes for a site and date range (with prefetch support).
#[pyfunction]
#[pyo3(name = "iter_volumes", signature = (site, start, end = None, schema = None, qc = None, fold_size = None, prefetch = None))]
pub fn iter_volumes_py(
    site: &str,
    start: &str,
    end: Option<&str>,
    schema: Option<&str>,
    qc: Option<bool>,
    fold_size: Option<usize>,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
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
    let prefetch = prefetch.unwrap_or(3).max(1);

    let state = VolumeIterState {
        site: site.to_string(),
        current_date: start_date,
        end_date,
        volumes: Vec::new(),
        index: 0,
        prefetch,
        in_flight: VecDeque::new(),
    };

    let iterator = VolumeIterator {
        state: Arc::new(std::sync::Mutex::new(state)),
        schema,
        fold_size,
        _qc_enabled: qc_enabled,
    };

    Python::attach(|py| {
        let obj = Py::new(py, iterator)?;
        let bound = obj.into_pyobject(py).unwrap();
        Ok(bound.into_any().unbind())
    })
}

/// Iterate over volumes asynchronously (prefetching enabled).
#[pyfunction]
#[pyo3(name = "iter_volumes_async", signature = (site, start, end = None, schema = None, qc = None, fold_size = None, prefetch = None))]
pub fn iter_volumes_async_py(
    site: &str,
    start: &str,
    end: Option<&str>,
    schema: Option<&str>,
    qc: Option<bool>,
    fold_size: Option<usize>,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
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
    let prefetch = prefetch.unwrap_or(5).max(1);

    let state = VolumeIterState {
        site: site.to_string(),
        current_date: start_date,
        end_date,
        volumes: Vec::new(),
        index: 0,
        prefetch,
        in_flight: VecDeque::new(),
    };

    let iterator = VolumeIteratorAsync {
        state: Arc::new(Mutex::new(state)),
        schema,
        fold_size,
        _qc_enabled: qc_enabled,
    };

    Python::attach(|py| {
        let obj = Py::new(py, iterator)?;
        let bound = obj.into_pyobject(py).unwrap();
        Ok(bound.into_any().unbind())
    })
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
