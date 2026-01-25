//! Metadata iteration over explicit candidate lists (URL or URL+VolumeInfo).

use crate::error::{RadrsError, Result};
use crate::fetch::RUNTIME;
use crate::iter::peek::peek_volume_with_mode;
use crate::iter::{VolumeInfo, VolumeMeta};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

#[derive(Clone)]
struct MetaRef {
    url: String,
    info: Option<VolumeInfo>,
}

struct MetaIterState {
    items: VecDeque<MetaRef>,
    prefetch: usize,
    header_only: bool,
    peek_mode: String,
    in_flight: VecDeque<JoinHandle<Result<(MetaRef, VolumeMeta)>>>,
}

impl MetaIterState {
    async fn fill_prefetch(&mut self) -> Result<()> {
        while self.in_flight.len() < self.prefetch {
            let Some(meta_ref) = self.items.pop_front() else {
                break;
            };

            let header_only = self.header_only;
            let peek_mode = self.peek_mode.clone();

            let handle = tokio::spawn(async move {
                let meta = peek_volume_with_mode(&meta_ref.url, header_only, &peek_mode).await?;
                Ok((meta_ref, meta))
            });

            self.in_flight.push_back(handle);
        }

        Ok(())
    }

    async fn next_meta(&mut self) -> Result<Option<(MetaRef, VolumeMeta)>> {
        if self.in_flight.is_empty() {
            self.fill_prefetch().await?;
        }

        while let Some(handle) = self.in_flight.pop_front() {
            let result = handle
                .await
                .map_err(|e| RadrsError::Python(format!("Peek task failed: {}", e)))?;
            match result {
                Ok(meta) => {
                    self.fill_prefetch().await?;
                    return Ok(Some(meta));
                }
                Err(e) => {
                    tracing::warn!("Failed to peek volume: {}", e);
                    self.fill_prefetch().await?;
                    continue;
                }
            }
        }

        Ok(None)
    }
}

#[pyclass]
struct MetaIterator {
    state: Arc<std::sync::Mutex<MetaIterState>>,
    return_info: bool,
}

#[pymethods]
impl MetaIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        loop {
            let next_item = py.detach(|| {
                let state = self.state.clone();
                RUNTIME.block_on(async move {
                    let mut guard = state.lock().expect("state lock poisoned");
                    guard.next_meta().await
                })
            })?;

            let Some((meta_ref, meta)) = next_item else {
                return Err(pyo3::exceptions::PyStopIteration::new_err(()));
            };

            if self.return_info {
                let info = meta_ref.info.ok_or_else(|| {
                    RadrsError::Parse("Missing VolumeInfo for candidate".to_string())
                })?;
                let info_obj = Py::new(py, info)?;
                let meta_obj = Py::new(py, meta)?;
                let tuple = (info_obj, meta_obj).into_pyobject_or_pyerr(py)?;
                return Ok(tuple.into_any().unbind());
            }

            let meta_obj = Py::new(py, meta)?;
            return Ok(meta_obj.into_any());
        }
    }
}

#[pyclass]
struct MetaIteratorAsync {
    state: Arc<Mutex<MetaIterState>>,
    return_info: bool,
}

#[pymethods]
impl MetaIteratorAsync {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = slf.state.clone();
        let return_info = slf.return_info;

        let awaitable = future_into_py(py, async move {
            loop {
                let next_item = {
                    let mut guard = state.lock().await;
                    guard.next_meta().await?
                };

                let Some((meta_ref, meta)) = next_item else {
                    return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(()));
                };

                return Python::attach(|py| {
                    if return_info {
                        let info = meta_ref.info.ok_or_else(|| {
                            RadrsError::Parse("Missing VolumeInfo for candidate".to_string())
                        })?;
                        let info_obj = Py::new(py, info)?;
                        let meta_obj = Py::new(py, meta)?;
                        let tuple = (info_obj, meta_obj).into_pyobject_or_pyerr(py)?;
                        return Ok::<Py<PyAny>, pyo3::PyErr>(tuple.into_any().unbind());
                    }

                    let meta_obj = Py::new(py, meta)?;
                    Ok::<Py<PyAny>, pyo3::PyErr>(meta_obj.into_any())
                })
                .map_err(Into::into);
            }
        })?;

        Ok(awaitable.into())
    }
}

fn validate_peek_mode(peek_mode: &str) -> Result<()> {
    if peek_mode == "fast" || peek_mode == "small" {
        Ok(())
    } else {
        Err(RadrsError::Parse(format!(
            "Invalid peek_mode: {peek_mode} (expected 'fast' or 'small')"
        )))
    }
}

fn build_state_from_urls(
    urls: Vec<String>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> Result<MetaIterState> {
    validate_peek_mode(peek_mode)?;
    let prefetch = prefetch.unwrap_or(8).max(1);
    Ok(MetaIterState {
        items: urls
            .into_iter()
            .map(|url| MetaRef { url, info: None })
            .collect(),
        prefetch,
        header_only,
        peek_mode: peek_mode.to_string(),
        in_flight: VecDeque::new(),
    })
}

fn build_state_from_candidates(
    py: Python<'_>,
    candidates: Vec<(String, Py<VolumeInfo>)>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> Result<MetaIterState> {
    validate_peek_mode(peek_mode)?;
    let prefetch = prefetch.unwrap_or(8).max(1);
    let items = candidates
        .into_iter()
        .map(|(url, info)| MetaRef {
            url,
            info: Some(info.borrow(py).clone()),
        })
        .collect();

    Ok(MetaIterState {
        items,
        prefetch,
        header_only,
        peek_mode: peek_mode.to_string(),
        in_flight: VecDeque::new(),
    })
}

/// Iterate over metadata for a list of URLs.
#[pyfunction]
#[pyo3(
    name = "iter_meta_urls",
    signature = (urls, header_only = false, peek_mode = "fast", prefetch = None)
)]
pub fn iter_meta_urls_py(
    py: Python<'_>,
    urls: Vec<String>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let state = build_state_from_urls(urls, header_only, peek_mode, prefetch)?;
    let iterator = MetaIterator {
        state: Arc::new(std::sync::Mutex::new(state)),
        return_info: false,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}

/// Iterate over metadata for a list of URLs asynchronously.
#[pyfunction]
#[pyo3(
    name = "iter_meta_urls_async",
    signature = (urls, header_only = false, peek_mode = "fast", prefetch = None)
)]
pub fn iter_meta_urls_async_py(
    py: Python<'_>,
    urls: Vec<String>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let state = build_state_from_urls(urls, header_only, peek_mode, prefetch)?;
    let iterator = MetaIteratorAsync {
        state: Arc::new(Mutex::new(state)),
        return_info: false,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}

/// Iterate over metadata for a list of (url, VolumeInfo) candidates.
#[pyfunction]
#[pyo3(
    name = "iter_meta_candidates",
    signature = (candidates, header_only = false, peek_mode = "fast", prefetch = None)
)]
pub fn iter_meta_candidates_py(
    py: Python<'_>,
    candidates: Vec<(String, Py<VolumeInfo>)>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let state = build_state_from_candidates(py, candidates, header_only, peek_mode, prefetch)?;
    let iterator = MetaIterator {
        state: Arc::new(std::sync::Mutex::new(state)),
        return_info: true,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}

/// Iterate over metadata for a list of (url, VolumeInfo) candidates asynchronously.
#[pyfunction]
#[pyo3(
    name = "iter_meta_candidates_async",
    signature = (candidates, header_only = false, peek_mode = "fast", prefetch = None)
)]
pub fn iter_meta_candidates_async_py(
    py: Python<'_>,
    candidates: Vec<(String, Py<VolumeInfo>)>,
    header_only: bool,
    peek_mode: &str,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let state = build_state_from_candidates(py, candidates, header_only, peek_mode, prefetch)?;
    let iterator = MetaIteratorAsync {
        state: Arc::new(Mutex::new(state)),
        return_info: true,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}
