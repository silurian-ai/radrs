from __future__ import annotations

import argparse
import json
from pathlib import Path

import radrs.xradar as rxr


DEFAULT_VCP_JSON = "notebooks/vcp_samples_0.json"

# Variables to chunk as full arrays (one chunk per sweep)
FULL_CHUNK_VARS = {
    # 2D moment variables
    "DBZH", "ZDR", "PHIDP", "RHOHV", "CCORH", "VRADH", "WRADH",
    # 1D coordinate variables
    "azimuth", "elevation", "time", "range",
}


def get_encoding_for_datatree(dt):
    """Build encoding dict for all variables in a DataTree.

    Format: {"group_path": {"variable": {"chunks": ...}}}

    Uses full array chunks (one chunk per sweep) for efficient cloud storage reads.
    This results in ~2-5 MB chunks for typical NEXRAD sweeps.
    """
    encoding = {}
    for path, node in dt.subtree_with_keys:
        if not hasattr(node, 'ds') or node.ds is None:
            continue
        # Use full path with leading /
        group_path = "/" if path == "." else f"/{path}"
        group_encoding = {}

        for var_name in node.ds.data_vars:
            if var_name in FULL_CHUNK_VARS:
                var = node.ds[var_name]
                # Full array as single chunk
                group_encoding[var_name] = {"chunks": tuple(var.shape)}

        for coord_name in node.ds.coords:
            if coord_name in FULL_CHUNK_VARS:
                coord = node.ds.coords[coord_name]
                if coord.ndim > 0:  # Skip scalar coords
                    group_encoding[coord_name] = {"chunks": tuple(coord.shape)}

        if group_encoding:
            encoding[group_path] = group_encoding
    return encoding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump full xradar DataTree per VCP to a new Zarr path",
    )
    parser.add_argument("--vcp-json", default=DEFAULT_VCP_JSON)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-optimize-chunks",
        action="store_true",
        help="Disable cloud-optimized chunking (use xarray defaults)",
    )

    args = parser.parse_args()

    vcp_path = Path(args.vcp_json)
    if not vcp_path.exists():
        raise SystemExit(f"Missing VCP list: {vcp_path}")

    with vcp_path.open() as f:
        vcp_samples = json.load(f)

    output_base = args.output_base.rstrip("/")

    for vcp_key in sorted(vcp_samples.keys(), key=lambda x: int(x)):
        sample = vcp_samples[vcp_key]
        url = sample.get("url")
        if not url:
            print(f"Skipping VCP {vcp_key}: missing url")
            continue

        vcp_name = f"VCP-{vcp_key}"
        output_path = f"{output_base}/{vcp_name}.zarr"

        print(f"\n{vcp_name}")
        print(f"  input : {url}")
        print(f"  output: {output_path}")

        if args.dry_run:
            continue

        dt = rxr.open_datatree(url, sort_by_azimuth=True)

        # Use cloud-optimized chunking by default
        if args.no_optimize_chunks:
            dt.to_zarr(output_path, mode="w")
        else:
            encoding = get_encoding_for_datatree(dt)
            dt.to_zarr(output_path, mode="w", encoding=encoding)

    print("Done.")


if __name__ == "__main__":
    main()
