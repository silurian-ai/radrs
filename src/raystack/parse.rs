//! Raystack parsing and shared metadata/QC helpers.
//!
//! Single-volume parsing now delegates to `RaystackBatchData` so all
//! raystack outputs share one chunked return representation.

use crate::error::{RadrsError, Result};
use crate::fetch::{RUNTIME, default_open_datatree_storage_options, fetch_bytes_from_url};
use crate::range::{RangeGeometry, canonical_lattice, geometry};
use crate::raystack::batch::{add_activity_to_dict, compute_batch_activity, parse_single_volume};
use nexrad_model::data::{DataMoment, Scan};
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use pyo3_async_runtimes::tokio::future_into_py;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

/// Default fold size for raystack format.
pub const DEFAULT_FOLD_SIZE: usize = 128;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Decompress outer gzip if present. Uses Cow to avoid allocation when not gzipped.
pub(crate) fn ungzip_if_needed(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    if data.starts_with(&GZIP_MAGIC) {
        let mut decoder: flate2::read::GzDecoder<&[u8]> = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(Cow::Owned(decompressed))
    } else {
        Ok(Cow::Borrowed(data))
    }
}

/// Sweep metadata collected in first pass.
#[derive(Clone, Copy, Default)]
pub struct SweepMeta {
    pub elevation_number: u8,
    pub elevation_angle: f32,
    pub n_radials: usize,
    pub max_gates: usize,
    pub range_first_km: f64,
    pub gate_interval_km: f64,
    pub min_time: i64,
    pub max_time: i64,
}

/// Volume metadata from first pass.
pub struct VolumeMeta {
    pub sweeps: Vec<SweepMeta>,
    pub min_time: i64,
    pub max_time: i64,
}

#[derive(Clone)]
pub enum QcOp {
    RhohvThreshold {
        threshold: f32,
        vname: String,
    },
    SunSpike {
        dbzh_threshold: f32,
        fill_threshold: f32,
        corr_threshold: f32,
        vname: String,
    },
    VradhWindingNumber {
        nyquist: Option<f32>,
        wind_size: usize,
        velocity_texture_threshold: f32,
        reflectivity_threshold: f32,
        interval_splits: usize,
        skip_between_rays: usize,
        skip_along_ray: usize,
        centered: bool,
        rays_wrap_around: bool,
        fill_value: Option<f32>,
        fill_tolerance: f32,
        vname: String,
    },
}

pub enum QcArray {
    Mask(Vec<i8>),
    Float(Vec<f32>),
}

pub(crate) fn parse_qc_ops(py: Python<'_>, qc: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<QcOp>> {
    let Some(qc_any) = qc else {
        return Ok(Vec::new());
    };

    if qc_any.is_none() {
        return Ok(Vec::new());
    }

    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = qc_any.cast::<PyList>() {
        list.iter().map(|item| item.to_owned()).collect()
    } else {
        vec![qc_any.to_owned()]
    };

    let mut ops = Vec::new();
    for item in items {
        if let Ok(name) = item.extract::<String>() {
            ops.push(qc_op_from_name(&name, None, py)?);
            continue;
        }

        if let Ok(dict) = item.cast::<PyDict>() {
            let name_any = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("QC spec missing 'name'"))?;
            let name: String = name_any.extract()?;
            ops.push(qc_op_from_name(&name, Some(dict), py)?);
            continue;
        }

        return Err(pyo3::exceptions::PyTypeError::new_err(
            "qc must be a QC spec or list of specs",
        ));
    }

    Ok(ops)
}

fn dict_f32(dict: &Bound<'_, PyDict>, key: &str, default: f32) -> PyResult<f32> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_string(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default.to_string());
        }
        return value.extract();
    }
    Ok(default.to_string())
}

fn dict_usize(dict: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(default);
        }
        return value.extract();
    }
    Ok(default)
}

fn dict_opt_f32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f32>> {
    if let Some(value) = dict.get_item(key)? {
        if value.is_none() {
            return Ok(None);
        }
        let v: f32 = value.extract()?;
        return Ok(Some(v));
    }
    Ok(None)
}

fn qc_op_from_name(
    name: &str,
    dict: Option<&Bound<'_, PyDict>>,
    _py: Python<'_>,
) -> PyResult<QcOp> {
    match name {
        "rhohv_threshold" => {
            let threshold = if let Some(dict) = dict {
                dict_f32(dict, "threshold", 0.8)?
            } else {
                0.8
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "rhohv_threshold_mask")?
            } else {
                "rhohv_threshold_mask".to_string()
            };
            Ok(QcOp::RhohvThreshold { threshold, vname })
        }
        "sun_spike" => {
            let dbzh_threshold = if let Some(dict) = dict {
                dict_f32(dict, "dbzh_threshold", 0.0)?
            } else {
                0.0
            };
            let fill_threshold = if let Some(dict) = dict {
                dict_f32(dict, "fill_threshold", 0.9)?
            } else {
                0.9
            };
            let corr_threshold = if let Some(dict) = dict {
                dict_f32(dict, "corr_threshold", 0.8)?
            } else {
                0.8
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "sun_spike_mask")?
            } else {
                "sun_spike_mask".to_string()
            };
            Ok(QcOp::SunSpike {
                dbzh_threshold,
                fill_threshold,
                corr_threshold,
                vname,
            })
        }
        "vradh_winding_number" => {
            let nyquist = if let Some(dict) = dict {
                dict_opt_f32(dict, "nyquist")?
            } else {
                None
            };
            let wind_size = if let Some(dict) = dict {
                dict_usize(dict, "wind_size", 3)?
            } else {
                3
            };
            let velocity_texture_threshold = if let Some(dict) = dict {
                dict_f32(dict, "velocity_texture_threshold", 4.0)?
            } else {
                4.0
            };
            let reflectivity_threshold = if let Some(dict) = dict {
                dict_f32(dict, "reflectivity_threshold", 0.0)?
            } else {
                0.0
            };
            let interval_splits = if let Some(dict) = dict {
                dict_usize(dict, "interval_splits", 3)?
            } else {
                3
            };
            let skip_between_rays = if let Some(dict) = dict {
                dict_usize(dict, "skip_between_rays", 100)?
            } else {
                100
            };
            let skip_along_ray = if let Some(dict) = dict {
                dict_usize(dict, "skip_along_ray", 100)?
            } else {
                100
            };
            let centered = if let Some(dict) = dict {
                dict_bool(dict, "centered", true)?
            } else {
                true
            };
            let rays_wrap_around = if let Some(dict) = dict {
                dict_bool(dict, "rays_wrap_around", true)?
            } else {
                true
            };
            let fill_value = if let Some(dict) = dict {
                dict_opt_f32(dict, "fill_value")?
            } else {
                Some(-64.5)
            };
            let fill_tolerance = if let Some(dict) = dict {
                dict_f32(dict, "fill_tolerance", 1.0)?
            } else {
                1.0
            };
            let vname = if let Some(dict) = dict {
                dict_string(dict, "vname", "vradh_winding_number")?
            } else {
                "vradh_winding_number".to_string()
            };
            Ok(QcOp::VradhWindingNumber {
                nyquist,
                wind_size,
                velocity_texture_threshold,
                reflectivity_threshold,
                interval_splits,
                skip_between_rays,
                skip_along_ray,
                centered,
                rays_wrap_around,
                fill_value,
                fill_tolerance,
                vname,
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown QC step: {}",
            name
        ))),
    }
}

fn collect_moment_geometry<M: DataMoment>(
    moment: Option<&M>,
    geometries: &mut Vec<RangeGeometry>,
) -> Result<()> {
    if let Some(moment) = moment {
        geometries.push(geometry(moment)?);
    }
    Ok(())
}

/// First pass: collect metadata without allocating moment data.
pub fn collect_metadata(scan: &Scan) -> Result<VolumeMeta> {
    let mut vcp_min_time = i64::MAX;
    let mut vcp_max_time = i64::MIN;
    let mut sweeps = Vec::new();

    for sweep in scan.sweeps() {
        let radials = sweep.radials();
        let n_radials = radials.len();
        if n_radials == 0 {
            continue;
        }

        let mut sweep_min_time = i64::MAX;
        let mut sweep_max_time = i64::MIN;

        let first_radial = &radials[0];

        let mut geometries = Vec::new();

        for radial in radials {
            collect_moment_geometry(radial.reflectivity(), &mut geometries)?;
            collect_moment_geometry(radial.velocity(), &mut geometries)?;
            collect_moment_geometry(radial.spectrum_width(), &mut geometries)?;
            collect_moment_geometry(radial.differential_reflectivity(), &mut geometries)?;
            collect_moment_geometry(radial.differential_phase(), &mut geometries)?;
            collect_moment_geometry(radial.correlation_coefficient(), &mut geometries)?;
            collect_moment_geometry(radial.clutter_filter_power(), &mut geometries)?;

            vcp_min_time = vcp_min_time.min(radial.collection_timestamp());
            vcp_max_time = vcp_max_time.max(radial.collection_timestamp());
            sweep_min_time = sweep_min_time.min(radial.collection_timestamp());
            sweep_max_time = sweep_max_time.max(radial.collection_timestamp());
        }

        let lattice = canonical_lattice(&geometries)?;
        let (max_gates, range_first_km, gate_interval_km) = lattice
            .map(|grid| {
                (
                    grid.gate_count,
                    grid.first_gate_m as f64 * 0.001,
                    grid.gate_spacing_m as f64 * 0.001,
                )
            })
            .unwrap_or((0, 0.0, 0.0));

        sweeps.push(SweepMeta {
            elevation_number: sweep.elevation_number(),
            elevation_angle: first_radial.elevation_angle_degrees(),
            n_radials,
            max_gates,
            range_first_km,
            gate_interval_km,
            min_time: sweep_min_time,
            max_time: sweep_max_time,
        });
    }

    Ok(VolumeMeta {
        sweeps,
        min_time: vcp_min_time,
        max_time: vcp_max_time,
    })
}

fn finalize_batch_to_dict(
    py: Python<'_>,
    mut batch: crate::raystack::batch::RaystackBatchData,
    qc_ops: &[QcOp],
    include_activity: bool,
) -> PyResult<Py<PyDict>> {
    if !qc_ops.is_empty() {
        batch.add_qc_outputs(qc_ops);
    }
    let activity = if include_activity {
        Some(compute_batch_activity(&batch))
    } else {
        None
    };

    let out = batch.to_python_dict(py)?;
    if let Some(activity) = activity {
        let out_bound = out.bind(py);
        add_activity_to_dict(py, out_bound, activity)?;
    }

    Ok(out)
}

/// Parse NEXRAD data to raystack format (Python wrapper).
#[pyfunction]
#[pyo3(name = "parse", signature = (data, fold_size = None, qc = None, include_activity = true))]
pub fn parse_py<'py>(
    py: Python<'py>,
    data: &[u8],
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;
    let bytes_len = data.len();
    let start = Instant::now();

    let batch = match py.detach(|| parse_single_volume(data, fold_size)) {
        Ok(batch) => batch,
        Err(err) => {
            tracing::warn!(
                target: "radrs::parse",
                format = "raystack",
                bytes = bytes_len,
                fold_size,
                error = %err
            );
            return Err(err.into());
        }
    };
    let elapsed = start.elapsed();
    let (_, _, n_sweeps, _, n_returns, _) = batch.progress();
    tracing::info!(
        target: "radrs::parse",
        format = "raystack",
        bytes = bytes_len,
        fold_size,
        n_returns,
        n_sweeps,
        elapsed_ms = elapsed.as_millis() as u64
    );

    Ok(finalize_batch_to_dict(py, batch, &qc_ops, include_activity)?.into())
}

/// Open a NEXRAD Level 2 file and return raystack DataTree.
#[pyfunction]
#[pyo3(name = "open_datatree", signature = (source, fold_size = None, qc = None, include_activity = true, storage_options = None))]
pub fn open_raystack_datatree_py<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;

    let data = if source.is_instance_of::<PyBytes>() {
        source.extract::<Vec<u8>>()?
    } else {
        let path_str: String = source.extract()?;
        let storage_options =
            storage_options.or_else(|| default_open_datatree_storage_options(&path_str));
        RUNTIME
            .block_on(fetch_bytes_from_url(&path_str, storage_options))?
            .to_vec()
    };

    let bytes_len = data.len();
    let start = Instant::now();
    let batch = match py.detach(|| parse_single_volume(&data, fold_size)) {
        Ok(batch) => batch,
        Err(err) => {
            tracing::warn!(
                target: "radrs::parse",
                format = "raystack",
                bytes = bytes_len,
                fold_size,
                error = %err
            );
            return Err(err.into());
        }
    };
    let elapsed = start.elapsed();
    let (_, _, n_sweeps, _, n_returns, _) = batch.progress();
    tracing::info!(
        target: "radrs::parse",
        format = "raystack",
        bytes = bytes_len,
        fold_size,
        n_returns,
        n_sweeps,
        elapsed_ms = elapsed.as_millis() as u64
    );

    let dict = finalize_batch_to_dict(py, batch, &qc_ops, include_activity)?;
    crate::raystack::convert::to_raystack_datatree_py(py, dict.bind(py))
}

/// Open a NEXRAD Level 2 file asynchronously and return raystack DataTree.
#[pyfunction]
#[pyo3(name = "open_datatree_async", signature = (source, fold_size = None, qc = None, include_activity = true, storage_options = None))]
pub fn open_raystack_datatree_async_py<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    fold_size: Option<usize>,
    qc: Option<&Bound<'py, PyAny>>,
    include_activity: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let source = source.as_borrowed().to_owned().unbind();
    let fold_size = fold_size.unwrap_or(DEFAULT_FOLD_SIZE);
    let qc_ops = parse_qc_ops(py, qc)?;

    let awaitable = future_into_py(py, async move {
        let data = fetch_source_bytes_async(source, storage_options).await?;

        let bytes_len = data.len();
        let start = Instant::now();
        let batch = tokio::task::spawn_blocking(move || parse_single_volume(&data, fold_size))
            .await
            .map_err(|e| RadrsError::Python(format!("Parse task failed: {}", e)))??;
        let elapsed = start.elapsed();
        let (_, _, n_sweeps, _, n_returns, _) = batch.progress();
        tracing::info!(
            target: "radrs::parse",
            format = "raystack",
            bytes = bytes_len,
            fold_size,
            n_returns,
            n_sweeps,
            elapsed_ms = elapsed.as_millis() as u64
        );

        Python::attach(|py| {
            let dict = finalize_batch_to_dict(py, batch, &qc_ops, include_activity)?;
            crate::raystack::convert::to_raystack_datatree_py(py, dict.bind(py))
        })
    })?;

    Ok(awaitable.into())
}

async fn fetch_source_bytes_async(
    source: Py<PyAny>,
    storage_options: Option<HashMap<String, String>>,
) -> Result<Vec<u8>> {
    enum Source {
        Bytes(Vec<u8>),
        Path(String),
    }

    let source = Python::attach(|py| -> Result<Source> {
        let obj = source.bind(py);
        if obj.is_instance_of::<PyBytes>() {
            let bytes = obj
                .extract::<Vec<u8>>()
                .map_err(|e: PyErr| RadrsError::Python(e.to_string()))?;
            Ok(Source::Bytes(bytes))
        } else {
            let path = obj
                .extract::<String>()
                .map_err(|e: PyErr| RadrsError::Python(e.to_string()))?;
            Ok(Source::Path(path))
        }
    })?;

    match source {
        Source::Bytes(bytes) => Ok(bytes),
        Source::Path(path) => {
            let storage_options =
                storage_options.or_else(|| default_open_datatree_storage_options(&path));
            fetch_bytes_from_url(&path, storage_options)
                .await
                .map(|b| b.to_vec())
        }
    }
}
