"""xradar-compatible interface.

Drop-in replacement for ``xradar.io.open_nexradlevel2_datatree`` —
returns an ``xarray.DataTree`` with the same structure as xradar.

Examples
--------
Open a NEXRAD volume from S3, a local path, or in-memory bytes:

```python
import radrs.xradar as rxr

dt = rxr.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000000_V06")
dt = rxr.open_datatree("/path/to/local/file.ar2v")
dt = rxr.open_datatree(file_bytes)
```

Cloud support
-------------
The URI fetch layer accepts any scheme `object_store` supports (`s3://`,
`gs://`, `az://`, `file://`, local paths) and forwards `storage_options` to
it — so a single-volume NEXRAD file living on any of these works.

For the public NEXRAD archive, AWS S3 (``s3://unidata-nexrad-level2``) is
the only cloud source that works out of the box: it stores one volume per
key. The public GCS mirror (``gs://gcp-public-data-nexrad-l2``) batches
volumes into 6-minute tar archives, which we do not unpack — pointing
``open_datatree`` at one of those URLs will fetch the bytes successfully
but fail at parse. We are not aware of a public Azure mirror at the time
of writing, so ``az://`` is plumbed but untested for NEXRAD.

Public buckets need ``storage_options={"anon": "true"}`` (or
``"skip_signature": "true"``); the same convention as
``radrs.NexradL2ArchiveIter``. Without it, the underlying ``object_store``
client tries the credential chain (env, config, IMDS) and fails on
unauthenticated machines.
"""

# Import from the Rust extension
try:
    from radrs._radrs import xradar as _xradar
except ImportError:
    _xradar = None

if _xradar is None:
    def open_datatree(source, sort_by_azimuth=False, storage_options=None):
        """Open a NEXRAD Level 2 file as xarray DataTree."""
        raise NotImplementedError("radrs.xradar module not available")

    async def open_datatree_async(source, sort_by_azimuth=False, storage_options=None):
        """Open a NEXRAD Level 2 file as xarray DataTree (async)."""
        raise NotImplementedError("radrs.xradar module not available")
else:
    def open_datatree(source, sort_by_azimuth=False, storage_options=None):
        """Open a NEXRAD Level 2 file as xarray DataTree.

        Parameters
        ----------
        source : str or bytes
            Path to file, URI (``s3://``, ``gs://``, ``az://``, ``file://``,
            or absolute local path), or raw bytes. The URI must point at a
            single NEXRAD Level 2 volume — tar archives (e.g. the public
            GCS NEXRAD mirror's 6-minute bundles) are not unpacked here and
            will fail at parse. See module docstring for details.
        sort_by_azimuth : bool, default=False
            If True, sort radials within each sweep by azimuth (xradar convention).
        storage_options : dict, optional
            Storage backend configuration forwarded to object_store. Examples:
            - S3 anonymous: {"anon": "true"}
            - GCS service account: {"service_account_path": "/path/to/key.json"}
            - Azure: {"account_name": "...", "access_key": "..."}
        """
        return _xradar.open_datatree(
            source,
            sort_by_azimuth=sort_by_azimuth,
            storage_options=storage_options,
        )

    async def open_datatree_async(source, sort_by_azimuth=False, storage_options=None):
        """Open a NEXRAD Level 2 file as xarray DataTree (async).

        See ``open_datatree`` for parameter descriptions, including the
        single-volume-only caveat for non-S3 cloud sources.
        """
        return await _xradar.open_datatree_async(
            source,
            sort_by_azimuth=sort_by_azimuth,
            storage_options=storage_options,
        )

__all__ = ["open_datatree", "open_datatree_async"]
