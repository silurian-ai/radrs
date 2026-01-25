#!/usr/bin/env python3
"""
Find sample volumes for each VCP type from the NEXRAD archive.

Uses peek_volume for efficient VCP extraction (~256KB per file instead of full download).

Known VCPs:
- Clear Air: 31 (long pulse), 32 (short pulse), 34 (newer), 35 (default since 2018)
- Precipitation: 12 (classic), 215 (default since 2018)
- Severe/Convective: 212 (SZ-2)
- Tropical: 112, 121 (velocity aliasing mitigation)
- Retired (May 2018): 11, 21, 211, 221
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import radrs

# Target sites and dates for different VCP types
SEARCH_TARGETS = [
    # Tornado alley - severe weather (VCP 212)
    ("KTLX", "2024-04-27"),  # Sulphur EF4 outbreak
    ("KTLX", "2024-05-06"),  # Barnsdall EF4 outbreak
    # Clear air (VCP 31, 32, 34, 35)
    ("KDVN", "2024-12-01"),  # Quad Cities winter - VCP 31
    ("KTLX", "2017-05-15"),  # Pre-2018 for VCP 32
    ("KFWS", "2025-01-10"),  # Dallas - VCP 34
    # Precipitation (VCP 12, 215)
    ("KFWS", "2025-01-10"),  # Dallas winter precip - VCP 12
    ("KATX", "2024-11-15"),  # Seattle - VCP 215
    # Tropical systems (VCP 112, 121)
    ("KTLH", "2023-08-30"),  # Hurricane Idalia - VCP 112
    ("KBRO", "2020-07-25"),  # Hurricane Hanna - VCP 121
    ("KMLB", "2022-09-28"),  # Hurricane Ian - VCP 112
    ("KLIX", "2021-08-29"),  # Hurricane Ida - VCP 112
    # Pre-2018 for retired VCPs (21, 221)
    ("KMPX", "2017-04-20"),  # Minneapolis spring rain - VCP 221
    ("KTLX", "2017-11-10"),  # Fall rain
]

# All known VCPs
# Note: 11, 21, 211, 221 were retired in May 2018
TARGET_VCPS = {12, 31, 32, 34, 35, 112, 121, 212, 215, 221}


def peek_with_url(site: str, date: str, volume_name: str) -> tuple[str, dict]:
    """Peek a volume and return (url, metadata)."""
    url = f"s3://unidata-nexrad-level2/{date.replace('-', '/')[:4]}/{date[5:7]}/{date[8:10]}/{site}/{volume_name}"
    try:
        meta = radrs.peek_volume(url)
        return url, {
            "vcp": meta.vcp,
            "site": meta.site,
            "datetime": str(meta.datetime),
            "version": meta.version,
            "file_size": meta.file_size,
            "latitude": meta.latitude,
            "longitude": meta.longitude,
            "moments": meta.moments,
        }
    except Exception as e:
        return url, {"error": str(e)}


def search_site_date(site: str, date: str, found_vcps: set, max_per_site: int = 50):
    """Search a site/date for VCPs, return list of (vcp, url, meta) for new VCPs found."""
    results = []
    volumes = radrs.list_volumes(site, date)

    if not volumes:
        print(f"  No volumes found for {site}/{date}")
        return results

    # Sample volumes evenly across the day
    step = max(1, len(volumes) // max_per_site)
    sample_volumes = volumes[::step][:max_per_site]

    print(f"  {site}/{date}: checking {len(sample_volumes)}/{len(volumes)} volumes...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(peek_with_url, site, date, v.name): v.name
            for v in sample_volumes
        }

        for future in as_completed(futures):
            url, meta = future.result()
            if "error" in meta:
                continue

            vcp = meta.get("vcp")
            if vcp is not None and vcp not in found_vcps:
                results.append((vcp, url, meta))
                found_vcps.add(vcp)
                print(f"    Found VCP {vcp}: {url}")

                # Early exit if we found all target VCPs
                if TARGET_VCPS <= found_vcps:
                    break

    return results


def main():
    parser = argparse.ArgumentParser(description="Find sample volumes for each VCP type")
    parser.add_argument(
        "--max-per-site", type=int, default=50, help="Max volumes to check per site/date"
    )
    parser.add_argument("--output", "-o", type=str, help="Output file for results (JSON)")
    args = parser.parse_args()

    print("Searching for VCP samples...")
    print(f"Target VCPs: {sorted(TARGET_VCPS)}")
    print()

    found_vcps: set[int] = set()
    all_results: list[tuple[int, str, dict]] = []

    for site, date in SEARCH_TARGETS:
        if TARGET_VCPS <= found_vcps:
            print("Found all target VCPs!")
            break

        results = search_site_date(site, date, found_vcps, args.max_per_site)
        all_results.extend(results)

    # Summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    by_vcp = {}
    for vcp, url, meta in all_results:
        by_vcp[vcp] = {"url": url, **meta}

    for vcp in sorted(by_vcp.keys()):
        info = by_vcp[vcp]
        print(f"\nVCP {vcp}:")
        print(f"  URL: {info['url']}")
        print(f"  Site: {info['site']}, Time: {info['datetime']}")
        print(f"  Moments: {info.get('moments')}")

    missing = TARGET_VCPS - set(by_vcp.keys())
    if missing:
        print(f"\nMissing VCPs: {sorted(missing)}")

    # Save to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(by_vcp, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
