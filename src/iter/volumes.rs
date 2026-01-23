//! Volume iteration over archive sources

use crate::error::{RadrsError, Result};
use crate::fetch::RUNTIME;
use crate::raystack::{self, parse_qc_ops, QcOp};
use crate::xradar;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::source::{VolumeSource, VolumeSourceState};

#[derive(Clone, Copy)]
enum OutputKind {
    Xradar,
    Raystack,
}

impl OutputKind {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("xradar") {
            "xradar" => Ok(OutputKind::Xradar),
            "raystack" => Ok(OutputKind::Raystack),
            other => Err(RadrsError::Parse(format!(
                "Invalid output: {other} (expected 'xradar' or 'raystack')"
            ))),
        }
    }
}

/// Sync iterator with prefetch support.
/// Uses the same VolumeIterState as the async iterator, driven via block_on.
#[pyclass]
struct VolumeIterator {
    state: Arc<std::sync::Mutex<VolumeIterState>>,
    output: OutputKind,
    fold_size: usize,
    qc_ops: Vec<QcOp>,
}

#[pymethods]
impl VolumeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Loop until we get a valid file or run out of files
        loop {
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

            // Use if/else to make ownership clear - data is moved into exactly one branch
            if matches!(self.output, OutputKind::Raystack) {
                let fold_size = self.fold_size;
                let parse_result = py.detach(|| raystack::parse_optimized(&data, fold_size));
                match parse_result {
                    Ok(raystack) => {
                        return raystack::raystack_to_python(py, raystack, &self.qc_ops, true);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse volume (raystack), skipping: {}", e);
                        continue;
                    }
                }
            } else {
                let parse_result = py.detach(|| xradar::open_datatree(data));
                match parse_result {
                    Ok((scan, meta)) => {
                        return xradar::datatree::scan_to_datatree(py, &scan, &meta);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse volume (xradar), skipping: {}", e);
                        continue;
                    }
                }
            }
        }
    }
}

struct VolumeIterState {
    source: VolumeSourceState,
    prefetch: usize,
    in_flight: VecDeque<JoinHandle<Result<Vec<u8>>>>,
}

impl VolumeIterState {
    async fn fill_prefetch(&mut self) -> Result<()> {
        while self.in_flight.len() < self.prefetch {
            let Some(volume_ref) = self.source.next_ref().await? else {
                break;
            };

            let handle = tokio::spawn(async move { volume_ref.fetch().await });
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
    output: OutputKind,
    fold_size: usize,
    qc_ops: Vec<QcOp>,
}

#[pymethods]
impl VolumeIteratorAsync {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = slf.state.clone();
        let output = slf.output;
        let fold_size = slf.fold_size;
        let qc_ops = slf.qc_ops.clone();

        let awaitable = future_into_py(py, async move {
            // Loop until we get a valid file or run out of files
            loop {
                let bytes = {
                    let mut guard = state.lock().await;
                    guard.next_bytes().await?
                };

                let Some(data) = bytes else {
                    return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(()));
                };

                // Use else to make ownership clear to the compiler - data is moved into exactly one branch
                if matches!(output, OutputKind::Raystack) {
                    let parse_result = tokio::task::spawn_blocking(move || {
                        raystack::parse_optimized(&data, fold_size)
                    })
                    .await
                    .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))?;

                    match parse_result {
                        Ok(raystack) => {
                            return Python::attach(|py| {
                                raystack::raystack_to_python(py, raystack, &qc_ops, true)
                            })
                            .map_err(Into::into);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse volume (raystack), skipping: {}", e);
                            continue;
                        }
                    }
                } else {
                    let parse_result =
                        tokio::task::spawn_blocking(move || xradar::open_datatree(data))
                            .await
                            .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))?;

                    match parse_result {
                        Ok((scan, meta)) => {
                            return Python::attach(|py| {
                                xradar::datatree::scan_to_datatree(py, &scan, &meta)
                            })
                            .map_err(Into::into);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse volume (xradar), skipping: {}", e);
                            continue;
                        }
                    }
                }
            }
        })?;

        Ok(awaitable.into())
    }
}

/// Iterate over volumes from a source (with prefetch support).
#[pyfunction]
#[pyo3(name = "iter_volumes", signature = (source, output = None, qc = None, fold_size = None, prefetch = None))]
pub fn iter_volumes_py(
    py: Python<'_>,
    source: PyRef<'_, VolumeSource>,
    output: Option<&str>,
    qc: Option<&Bound<'_, PyAny>>,
    fold_size: Option<usize>,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let output = OutputKind::parse(output)?;
    let fold_size = fold_size.unwrap_or(128);
    let qc_ops = parse_qc_ops(py, qc)?;
    let prefetch = prefetch.unwrap_or(3).max(1);

    let state = VolumeIterState {
        source: source.to_state(),
        prefetch,
        in_flight: VecDeque::new(),
    };

    let iterator = VolumeIterator {
        state: Arc::new(std::sync::Mutex::new(state)),
        output,
        fold_size,
        qc_ops,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}

/// Iterate over volumes asynchronously from a source (prefetching enabled).
#[pyfunction]
#[pyo3(name = "iter_volumes_async", signature = (source, output = None, qc = None, fold_size = None, prefetch = None))]
pub fn iter_volumes_async_py(
    py: Python<'_>,
    source: PyRef<'_, VolumeSource>,
    output: Option<&str>,
    qc: Option<&Bound<'_, PyAny>>,
    fold_size: Option<usize>,
    prefetch: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let output = OutputKind::parse(output)?;
    let fold_size = fold_size.unwrap_or(128);
    let qc_ops = parse_qc_ops(py, qc)?;
    let prefetch = prefetch.unwrap_or(5).max(1);

    let state = VolumeIterState {
        source: source.to_state(),
        prefetch,
        in_flight: VecDeque::new(),
    };

    let iterator = VolumeIteratorAsync {
        state: Arc::new(Mutex::new(state)),
        output,
        fold_size,
        qc_ops,
    };

    let obj = Py::new(py, iterator)?;
    let bound = obj.into_pyobject(py).unwrap();
    Ok(bound.into_any().unbind())
}
