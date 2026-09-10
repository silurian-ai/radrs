"""Raystack format tools for ML training.

This module provides tools for converting NEXRAD data into the raystack
format — a flat layout of `vcps`, `sweeps`, `returns`, and `activity` arrays
optimized for machine learning training pipelines.

Returns are chunked segments of physical radials. A single radial can produce
multiple returns when ``n_gates > fold_size``. Activity metrics are computed
over these return chunks.

Examples
--------
Open a NEXRAD volume as a raystack DataTree:

```python
import radrs.raystack as rrs

rdt = rrs.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06", fold_size=128)
rdt["returns"]["DBZH"].shape  # (n_returns, 128)
```

Convert an existing xradar DataTree to raystack:

```python
import radrs.xradar as rxr
import radrs.raystack as rrs

dt = rxr.open_datatree(file_path)
rs = rrs.from_xradar_datatree(dt, fold_size=128)
```

Apply built-in QC at parse time:

```python
import radrs.raystack as rrs
import radrs.qc as qc

rdt = rrs.open_datatree(src, fold_size=128, qc=[qc.RhohvThreshold(threshold=0.8)])
```

For lower-level use over already-loaded bytes, see ``parse``.
"""

import numpy as np
import xarray as xr

from radrs._source_format import DEFAULT_SOURCE_FORMAT, require_supported_format
from radrs.qc import compile_qc_steps

# Import from the Rust extension
from radrs._radrs import raystack as _raystack

def parse(
    data,
    fold_size=None,
    qc=None,
    include_activity=True,
    *,
    format: str = DEFAULT_SOURCE_FORMAT,
):
    require_supported_format(format)
    qc_spec = compile_qc_steps(qc)
    return _raystack.parse(
        data, fold_size=fold_size, qc=qc_spec, include_activity=include_activity
    )

def from_xradar_datatree(datatree, fold_size=None, include_activity=True):
    return _raystack.from_xradar_datatree(
        datatree, fold_size=fold_size, include_activity=include_activity
    )

to_xradar_datatree = _raystack.to_xradar_datatree
to_raystack_datatree = _raystack.to_raystack_datatree

def open_datatree(
    source,
    fold_size=None,
    qc=None,
    include_activity=True,
    storage_options=None,
    *,
    format: str = DEFAULT_SOURCE_FORMAT,
):
    """Open a NEXRAD volume and return a raystack DataTree.

    Parameters
    ----------
    source : str or bytes
        Path to file, URI (``s3://``, ``gs://``, ``az://``, ``file://``,
        or absolute local path), or raw bytes. The URI must point at a
        single NEXRAD Level 2 volume — tar archives (e.g. the public
        GCS NEXRAD mirror's 6-minute bundles) are not unpacked here and
        will fail at parse.
    fold_size : int, optional
        Range fold size (chunks each radial into segments).
    qc : list of QCStep or QCStep, optional
        Quality control steps to apply during parse.
    include_activity : bool, default=True
        Include activity metrics in the output.
    storage_options : dict, optional
        Storage backend configuration forwarded to object_store. Examples:
        - S3 anonymous: {"anon": "true"}
        - GCS service account: {"service_account_path": "/path/to/key.json"}
        - Azure: {"account_name": "...", "access_key": "..."}
    """
    require_supported_format(format)
    qc_spec = compile_qc_steps(qc)
    return _raystack.open_datatree(
        source,
        fold_size=fold_size,
        qc=qc_spec,
        include_activity=include_activity,
        storage_options=storage_options,
    )

async def open_datatree_async(
    source,
    fold_size=None,
    qc=None,
    include_activity=True,
    storage_options=None,
    *,
    format: str = DEFAULT_SOURCE_FORMAT,
):
    """Async variant of ``open_datatree``.

    See ``open_datatree`` for parameter descriptions, including the
    single-volume-only caveat for non-S3 cloud sources.
    """
    require_supported_format(format)
    qc_spec = compile_qc_steps(qc)
    return await _raystack.open_datatree_async(
        source,
        fold_size=fold_size,
        qc=qc_spec,
        include_activity=include_activity,
        storage_options=storage_options,
    )

class BatchedRaystack:
    """Pre-allocated raystack batch accumulator for incremental filling.

    Pre-allocates flat arrays for P patterns (VCPs), S sweeps, and R returns,
    then fills incrementally from volumes added to the batch. Each return tracks
    both its parent sweep_time and vcp_time for temporal organization.

    Parameters
    ----------
    max_vcps : int
        Maximum number of VCP patterns
    max_sweeps : int
        Maximum total number of sweeps across all patterns
    max_returns : int
        Maximum total number of returns/radials
    fold_size : int, default=128
        Range fold size
    truncate : bool, default=True
        If True, output arrays are truncated to actual filled size.
        If False, output arrays remain at max_returns size with NaN/0 fill.
    drop_empty_returns : bool, default=False
        If True, returns with all NaN moment data are excluded from the output.
        If False, all returns are included regardless of data completeness.
    include_sweeps : bool, default=True
        If True, sweep-level metadata is included in the output.
        If False, only VCP metadata is included.
    include_returns : bool, default=True
        If True, return-level data (radials and moment data) are included in the output.
        If False, only VCP and sweep metadata are collected.

    Examples
    --------
    Accumulate volumes from an L2 archive iterator into a batch:

    >>> import radrs
    >>> from datetime import datetime, timezone
    >>> # Allocate for 10 VCPs, ~140 sweeps (14 per VCP), ~50k returns
    >>> batch = radrs.raystack.BatchedRaystack(
    ...     max_vcps=10,
    ...     max_sweeps=140,
    ...     max_returns=50000,
    ...     fold_size=128,
    ... )
    >>> archive = radrs.NexradL2ArchiveIter(
    ...     base_uri="s3://unidata-nexrad-level2",
    ...     start_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
    ...     end_time=datetime(2024, 3, 16, tzinfo=timezone.utc),
    ...     storage_options={"anon": "true"},
    ...     site_filter=["KTLX"],
    ... )
    >>> n_added = batch.add_volumes_from_l2(archive, prefetch=8)
    >>> prog = batch.progress()
    >>> print(f"Collected {prog['patterns_filled']} patterns, "
    ...       f"{prog['sweeps_filled']} sweeps, {prog['returns_filled']} returns")

    Convert to DataTree for xarray:

    >>> datatree = batch.finalize_to_rs_dt()
    >>> datatree.to_zarr("batch_10vcps.zarr")

    Pre-allocate fixed-size arrays (no truncation):

    >>> # Request 75000 returns, get exactly 75000-element arrays
    >>> batch = radrs.raystack.BatchedRaystack(
    ...     max_vcps=10,
    ...     max_sweeps=140,
    ...     max_returns=75000,
    ...     fold_size=128,
    ...     truncate=False,  # Keep full size with NaN fill
    ... )
    >>> n_added = batch.add_volumes_from_l2(archive, prefetch=8)
    >>> raystack = batch.finalize_to_dict()
    >>> print(raystack['returns']['azimuth'].shape)  # (75000,) regardless of actual fills
    """

    def __init__(
        self,
        max_vcps,
        max_sweeps,
        max_returns,
        fold_size=128,
        truncate=True,
        drop_empty_returns=False,
        include_sweeps=True,
        include_returns=True,
    ):
        """Initialize batched raystack accumulator.

        Parameters
        ----------
        max_vcps : int
            Maximum number of VCP patterns
        max_sweeps : int
            Maximum total number of sweeps
        max_returns : int
            Maximum total number of returns
        fold_size : int, default=128
            Range fold size
        truncate : bool, default=True
            If True, output arrays are truncated to actual filled size.
            If False, output arrays remain at max_returns size with NaN/0 fill.
        drop_empty_returns : bool, default=False
            If True, returns with all NaN moment data are excluded from the output.
            If False, all returns are included regardless of data completeness.
        include_sweeps : bool, default=True
            If True, sweep-level metadata is included in the output.
            If False, only VCP metadata is included.
        include_returns : bool, default=True
            If True, return-level data (radials and moment data) are included in the output.
            If False, only VCP and sweep metadata are collected.
        """
        self._inner = _raystack.BatchedRaystack(
            max_vcps=max_vcps,
            max_sweeps=max_sweeps,
            max_returns=max_returns,
            fold_size=fold_size,
            truncate=truncate,
            drop_empty_returns=drop_empty_returns,
            include_sweeps=include_sweeps,
            include_returns=include_returns,
        )

    def add_volume(self, data):
        """Add a complete volume from raw bytes.

        Processes the volume directly from raw NEXRAD file bytes,
        avoiding intermediate dict allocation for efficiency.

        Parameters
        ----------
        data : bytes
            Raw NEXRAD volume file bytes

        Raises
        ------
        RuntimeError
            If capacity is exceeded or batch is finalized

        Examples
        --------
        >>> stream = radrs.raystack.BatchedRaystack(10, 140, 50000)
        >>> with open("/path/to/KTLX20240315_120000_V06", "rb") as f:
        ...     stream.add_volume(f.read())
        """
        return self._inner.add_volume(data)

    def add_volume_from_url(self, url, storage_options=None):
        """Add a volume from a cloud or local URL.

        Fetches the volume from the specified URL and adds it to the batch.
        Supports S3, GCS, Azure Blob Storage, and local filesystem URLs.

        Parameters
        ----------
        url : str
            Full URL to the volume file. Supported schemes:
            - S3: s3://bucket/path/to/file
            - GCS: gs://bucket/path/to/file
            - Azure: az://container/path/to/file or azure://container/path/to/file
            - Local: /path/to/file or file:///path/to/file
        storage_options : dict, optional
            Storage backend configuration. Examples:
            - S3: {"region": "us-east-1", "anon": "true"}
            - GCS: {"service_account_path": "/path/to/key.json"}
            - Azure: {"account_name": "...", "access_key": "..."}

        Raises
        ------
        RuntimeError
            If capacity is exceeded, batch is finalized, or fetch fails

        Examples
        --------
        S3 with anonymous access:

        >>> stream = radrs.raystack.BatchedRaystack(10, 140, 50000)
        >>> stream.add_volume_from_url(
        ...     "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06",
        ...     storage_options={"anon": "true"}
        ... )

        GCS with service account:

        >>> stream.add_volume_from_url(
        ...     "gs://my-bucket/nexrad/data/KTLX20240315_120000_V06",
        ...     storage_options={"service_account_path": "/path/to/key.json"}
        ... )

        Local filesystem:

        >>> stream.add_volume_from_url("/data/nexrad/KTLX20240315_120000_V06")

        Azure Blob Storage:

        >>> stream.add_volume_from_url(
        ...     "az://container/nexrad/KTLX20240315_120000_V06",
        ...     storage_options={"account_name": "myaccount", "access_key": "..."}
        ... )
        """
        return self._inner.add_volume_from_url(url, storage_options)

    def add_volumes_from_l2(self, l2_iter, prefetch=2):
        """Add volumes from a NexradL2ArchiveIter with prefetch support.

        This method supports multi-cloud sources (S3, GCS, Azure, local filesystem)
        using the NexradL2ArchiveIter which understands NEXRAD L2 Archive directory
        structure (YYYY/MM/DD/SITE/). Only volumes with timestamps within the
        specified time range are returned.

        Parameters
        ----------
        l2_iter : NexradL2ArchiveIter
            L2 archive iterator configured with time bounds
        prefetch : int, default=2
            Number of volumes to prefetch in parallel

        Returns
        -------
        int
            Number of volumes successfully added

        Raises
        ------
        RuntimeError
            If batch is already finalized

        Notes
        -----
        Running out of capacity is not an error. A volume that does not fit
        is rejected whole (adds are atomic, so a partial volume never lands)
        and iteration continues, so a later, smaller volume can still be
        admitted: an undersized batch quietly yields a time range with
        holes in it rather than a clean prefix. Iteration stops early only
        once a dimension is exactly full. Compare the returned count
        against the number of volumes you expected.

        Fetch failures are skipped the same way, as are parse failures on
        the default ``include_sweeps=True`` path; set ``RADRS_LOG=warn`` to
        see the reason for each skip. With ``include_sweeps=False``, a
        volume that cannot be peeked raises instead of being skipped.

        Examples
        --------
        S3 with anonymous access (time range within a single day):

        >>> import radrs
        >>> import radrs.raystack as rrs
        >>> from datetime import datetime, timezone
        >>> l2_iter = radrs.NexradL2ArchiveIter(
        ...     base_uri="s3://noaa-nexrad-level2",
        ...     start_time=datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc),
        ...     end_time=datetime(2024, 3, 15, 14, 0, 0, tzinfo=timezone.utc),
        ...     storage_options={"anon": "true"},
        ...     site_filter=["KTLX"]
        ... )
        >>> stream = rrs.BatchedRaystack(10, 140, 50000)
        >>> n_added = stream.add_volumes_from_l2(l2_iter, prefetch=8)

        Time range spanning multiple days:

        >>> l2_iter = radrs.NexradL2ArchiveIter(
        ...     base_uri="s3://noaa-nexrad-level2",
        ...     start_time=datetime(2024, 3, 15, 20, 0, 0, tzinfo=timezone.utc),
        ...     end_time=datetime(2024, 3, 16, 4, 0, 0, tzinfo=timezone.utc),
        ...     storage_options={"anon": "true"},
        ...     site_filter=["KTLX"]
        ... )

        GCS with service account:

        >>> l2_iter = radrs.NexradL2ArchiveIter(
        ...     base_uri="gs://my-bucket/nexrad",
        ...     start_time=datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc),
        ...     end_time=datetime(2024, 3, 16, tzinfo=timezone.utc),
        ...     storage_options={"service_account_path": "/path/to/key.json"}
        ... )
        >>> n_added = stream.add_volumes_from_l2(l2_iter, prefetch=8)

        Local filesystem:

        >>> l2_iter = radrs.NexradL2ArchiveIter(
        ...     base_uri="/data/nexrad",
        ...     start_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
        ...     end_time=datetime(2024, 3, 16, tzinfo=timezone.utc)
        ... )
        >>> n_added = stream.add_volumes_from_l2(l2_iter, prefetch=8)
        """
        return self._inner.add_volumes_from_l2(l2_iter, prefetch=prefetch)

    def progress(self):
        """Get current fill progress.

        Call this before ``finalize_to_dict()`` or ``finalize_to_rs_dt()``,
        which consume the batch — ``progress()`` panics once the buffers
        have been handed to Python. Plain ``finalize()`` leaves these
        figures untouched.

        Notes
        -----
        ``returns_filled`` counts the returns that survived
        ``drop_empty_returns``, while ``returns_capacity`` is checked against
        the uncompacted count. With compaction on, ``fill_fraction`` is
        therefore a lower bound on real capacity pressure — use the volume
        count returned by ``add_volumes_from_l2`` to confirm everything fit.

        Returns
        -------
        dict
            Progress with keys: patterns_filled, patterns_capacity,
            sweeps_filled, sweeps_capacity, returns_filled,
            returns_capacity, fill_fraction
        """
        return self._inner.progress()

    def has_capacity(self):
        """Check if batch has remaining capacity.

        Returns
        -------
        bool
            True if batch can accept more data
        """
        return self._inner.has_capacity()

    def add_qc_outputs(self, qc_steps):
        """Add quality control outputs to the batch.

        QC operations are computed over the data that has been accumulated
        so far. Call this before finalize() to include QC outputs in the
        final result.

        Parameters
        ----------
        qc_steps : list of QCStep or QCStep
            Quality control steps to apply. Examples:
            - radrs.qc.RhohvThreshold(threshold=0.8)
            - radrs.qc.SunSpike(dbzh_threshold=50.0)
            - radrs.qc.VradhWindingNumber(nyquist=25.0)

        Examples
        --------
        >>> import radrs.raystack as rrs
        >>> import radrs.qc as qc
        >>> batch = rrs.BatchedRaystack(10, 140, 50000)
        >>> # ... add volumes ...
        >>> batch.add_qc_outputs([
        ...     qc.RhohvThreshold(threshold=0.8, vname="rhohv_mask"),
        ...     qc.SunSpike(vname="sun_spike")
        ... ])
        >>> result = batch.finalize_to_dict()
        >>> # QC outputs will be in result['returns'] with 'qc.' prefix
        >>> print(result['returns']['qc.rhohv_mask'].shape)
        """
        qc_spec = compile_qc_steps(qc_steps)
        if qc_spec:
            return self._inner.add_qc_outputs(qc_spec)

    def finalize(self):
        """Finalize the batch in place.

        With the default ``truncate=True`` the arrays are already at their
        filled size and this is a no-op; with ``truncate=False`` they are
        padded out to full capacity with NaN/NaT fill. Either way the
        ``progress()`` counts are unchanged.

        Called automatically by finalize_to_dict() and finalize_to_rs_dt().
        """
        return self._inner.finalize()

    def finalize_to_dict(self):
        """Returns the batch as a raystack dict with all data zero-copied into python arrays.

        Returns
        -------
        dict
            Raystack dict with keys: vcps, sweeps, returns.
            Returns dict includes: vcp_time, sweep_time (parent times)
        """
        return self._inner.finalize_to_dict()

    def finalize_to_rs_dt(self):
        """Convert to xarray Raystack DataTree.

        Returns
        -------
        xarray.DataTree
            Raystack DataTree with hierarchical structure
        """

        rs_dict = self.finalize_to_dict()
        dt_dict = {}

        vcps_dict = rs_dict["vcps"]

        def _astype(arr: np.ndarray, dtype):
            # NOTE that we use the vectorized conversion here - ideally we would not
            # need to convert at all and raystacks would be datetime[ms] by convention
            if dtype == "datetime64[ms->ns]":
                return arr.astype("datetime64[ms]").astype("datetime64[ns]")
            elif dtype == "timedelta64[ms->ns]":
                return arr.astype("timedelta64[ms]").astype("timedelta64[ns]")
            else:
                return arr if dtype is None else arr.astype(dtype)

        vcps_ds = xr.Dataset(
            data_vars={
                dv: xr.Variable(["vcp_time"], _astype(vcps_dict[dv], dtype))
                for dv, dtype in [
                    ("source_fs_size", None),
                    ("instrument_name", None),
                    ("instrument_type", None),
                    ("platform_type", None),
                    ("latitude", None),
                    ("longitude", None),
                    ("altitude", None),
                    ("vcp_name", None),
                    ("vcp_number", None),
                    ("vcp_duration", "timedelta64[ms->ns]"),
                    ("num_sweeps", None),
                ]
            },
            coords={
                c: xr.Variable([c], _astype(vcps_dict[c], dtype))
                for c, dtype in [("vcp_time", "datetime64[ms->ns]")]
            },
        )

        dt_dict["vcps"] = vcps_ds

        if "sweeps" in rs_dict:
            sweeps_dict = rs_dict["sweeps"]

            sweeps_ds = xr.Dataset(
                data_vars={
                    dv: xr.Variable(["sweep_time"], _astype(sweeps_dict[dv], dtype))
                    for dv, dtype in [
                        ("vcp_time", "datetime64[ms->ns]"),
                        ("sweep_number", None),
                        ("sweep_duration", "timedelta64[ms->ns]"),
                        ("elevation_angle", None),
                        ("elevation_number", None),
                        ("range_start", None),
                        ("range_step", None),
                        ("max_range", None),
                        ("max_gates", None),
                        ("num_returns", None),
                    ]
                },
                coords={
                    c: xr.Variable([c], _astype(sweeps_dict[c], dtype))
                    for c, dtype in [("sweep_time", "datetime64[ms->ns]")]
                },
            )
            # Aliases
            sweeps_ds = sweeps_ds.assign(
                sweep_fixed_angle=sweeps_ds.variables["elevation_angle"]
            )

            dt_dict["sweeps"] = sweeps_ds

        if "returns" in rs_dict:
            returns_dict = rs_dict["returns"]
            moments_shape = (
                len(returns_dict["return_time"]),
                len(returns_dict["range"]),
            )
            qc_vars = [v for v in returns_dict if v.startswith("qc.")]

            returns_ds = xr.Dataset(
                data_vars=dict(
                    **{
                        dv: xr.Variable(
                            ["return_time"], _astype(returns_dict[dv], dtype)
                        )
                        for dv, dtype in [
                            ("vcp_time", "datetime64[ms->ns]"),
                            ("sweep_number", None),
                            ("sweep_time", "datetime64[ms->ns]"),
                            ("base_range", None),
                            ("range_step", None),
                            ("azimuth", None),
                            ("elevation", None),
                        ]
                    },
                    **{
                        dv: xr.Variable(
                            ["return_time", "range"],
                            _astype(returns_dict[dv], dtype).reshape(moments_shape),
                        )
                        for dv, dtype in [
                            ("DBZH", None),
                            ("VRADH", None),
                            ("WRADH", None),
                            ("ZDR", None),
                            ("PHIDP", None),
                            ("RHOHV", None),
                            ("CCORH", None),
                        ]
                        + [(qcv, None) for qcv in qc_vars]
                    },
                ),
                coords={
                    c: xr.Variable([c], _astype(returns_dict[c], dtype))
                    for c, dtype in [
                        ("return_time", "datetime64[ms->ns]"),
                        ("range", None),
                    ]
                },
            )

            dt_dict["returns"] = returns_ds

        return xr.DataTree.from_dict(dt_dict)

    def __repr__(self):
        return self._inner.__repr__()


__all__ = [
    "parse",
    "from_xradar_datatree",
    "to_xradar_datatree",
    "to_raystack_datatree",
    "open_datatree",
    "open_datatree_async",
    "BatchedRaystack",
]
