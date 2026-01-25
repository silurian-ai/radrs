from __future__ import annotations

import argparse
import json
from pathlib import Path

import radrs.xradar as rxr


DEFAULT_VCP_JSON = "notebooks/vcp_samples_0.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump full xradar DataTree per VCP to a new Zarr path",
    )
    parser.add_argument("--vcp-json", default=DEFAULT_VCP_JSON)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--dry-run", action="store_true")

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
        dt.to_zarr(output_path, mode="w")

    print("Done.")


if __name__ == "__main__":
    main()
