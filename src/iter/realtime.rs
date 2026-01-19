//! Streaming of NEXRAD data
//!
//! This module provides two streaming modes:
//!
//! 1. **Archive polling** (`stream_archive`): Polls the NOAA NEXRAD Level 2 archive
//!    bucket for new complete volumes. Has ~5 minute delay but provides complete volumes.
//!
//! 2. **Realtime chunks** (`stream_realtime`): Would stream from the Unidata chunks
//!    bucket for near-realtime data. Requires chunk accumulation (not yet implemented).

use crate::error::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

/// Stream NEXRAD data by polling the archive bucket (Python async generator)
///
/// Polls the NOAA NEXRAD Level 2 archive bucket for new complete volumes.
/// This has approximately 5 minute delay from real-time but provides complete volumes.
///
/// For true near-realtime streaming, see `stream_realtime` (not yet implemented).
///
/// # Arguments
/// * `site` - NEXRAD site ID (e.g., "KTLX", "KDMX")
/// * `poll_interval` - Seconds between polls (default: 30)
///
/// # Returns
/// Async generator yielding xarray DataTree objects
#[pyfunction]
#[pyo3(name = "stream_archive", signature = (site, poll_interval = None))]
pub fn stream_archive_py(py: Python<'_>, site: &str, poll_interval: Option<u64>) -> PyResult<Py<PyAny>> {
    let poll_interval = poll_interval.unwrap_or(30);

    // Create an async generator class in Python that polls for complete volumes
    let generator_code = r#"
import asyncio
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import re

class ArchiveStream:
    def __init__(self, site, poll_interval):
        self.site = site.upper()
        self.poll_interval = poll_interval
        self.seen = set()  # Only contains successfully processed volumes
        # Use the archive bucket with complete volumes
        self.bucket = "unidata-nexrad-level2"
        self._radrs = __import__('radrs')
        # Pattern to match NEXRAD Level 2 volume files (e.g., KTLX20240315_000000_V06)
        self._volume_pattern = re.compile(r'_V\d+$')

    def _is_volume_file(self, key):
        """Check if key is a NEXRAD volume file (not metadata or other files)."""
        # Volume files end with _V06, _V03, etc.
        return bool(self._volume_pattern.search(key))

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            # Get today's date for the prefix
            now = datetime.now(timezone.utc)
            prefix = f"{now.year}/{now.month:02d}/{now.day:02d}/{self.site}/"

            list_url = f"https://{self.bucket}.s3.amazonaws.com/?prefix={prefix}&list-type=2"

            try:
                with urllib.request.urlopen(list_url, timeout=10) as response:
                    content = response.read().decode('utf-8')
                    root = ET.fromstring(content)

                    # Find all Key elements (handle S3 namespace)
                    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                    keys = root.findall('.//s3:Key', ns)
                    if not keys:
                        keys = root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Key')

                    # Filter to volume files only, sort by key (timestamp) descending
                    key_texts = sorted(
                        [k.text for k in keys if k.text and self._is_volume_file(k.text)],
                        reverse=True
                    )

                    for key in key_texts:
                        if key in self.seen:
                            continue

                        # Fetch the volume
                        volume_url = f"https://{self.bucket}.s3.amazonaws.com/{key}"
                        try:
                            with urllib.request.urlopen(volume_url, timeout=60) as vol_response:
                                data = vol_response.read()
                                # Try to parse - only mark as seen if successful
                                dt = self._radrs.xradar.open_datatree(data)
                                # Success! Mark as seen and return
                                self.seen.add(key)
                                return dt
                        except Exception as e:
                            # Failed to fetch or parse - don't mark as seen, retry later
                            continue

            except Exception as e:
                # Listing failed - will retry on next poll
                pass

            await asyncio.sleep(self.poll_interval)

ArchiveStream
"#;

    let globals = PyDict::new(py);
    let code = CString::new(generator_code).unwrap();
    py.run(code.as_c_str(), Some(&globals), None)?;

    let stream_class = globals.get_item("ArchiveStream")?.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to create ArchiveStream class")
    })?;

    let stream = stream_class.call1((site, poll_interval))?;

    Ok(stream.into())
}

/// Stream realtime NEXRAD data (not yet implemented)
///
/// This would stream from the Unidata chunks bucket for near-realtime data.
/// Requires accumulating chunks into complete volumes before parsing.
///
/// For now, use `stream_archive` which polls for complete volumes with ~5 min delay.
#[pyfunction]
#[pyo3(name = "stream_realtime")]
pub fn stream_realtime_py(_py: Python<'_>, _site: &str) -> PyResult<Py<PyAny>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Realtime chunk streaming not yet implemented. \
         Use radrs.stream_archive() for archive polling with ~5 min delay, \
         or implement chunk accumulation for true realtime."
    ))
}

/// Stream NEXRAD data from archive (Rust async version - internal)
pub async fn stream_archive(_site: &str) -> Result<()> {
    unimplemented!("Use stream_archive_py for Python interface")
}
