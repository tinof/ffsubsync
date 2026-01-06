#!/usr/bin/env python3
"""
Verification script to calculate the offset between synced and reference subtitle files.
This helps determine if the +55s drift has been fixed.
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import srt


def parse_srt_file(filepath: str) -> list:
    """Parse an SRT file and return a list of subtitle objects."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    return list(srt.parse(content))


def calculate_offsets(synced_file: str, reference_file: str) -> dict:
    """
    Calculate timing offsets between synced and reference subtitle files.

    Returns a dictionary with statistics about the offset.
    """
    synced_subs = parse_srt_file(synced_file)
    ref_subs = parse_srt_file(reference_file)

    if len(synced_subs) != len(ref_subs):
        print(
            f"WARNING: Different number of subtitles: {len(synced_subs)} vs {len(ref_subs)}"
        )
        print("Using the minimum length for comparison")

    min_len = min(len(synced_subs), len(ref_subs))
    offsets = []

    for i in range(min_len):
        # Calculate offset in seconds (synced - reference)
        synced_start = synced_subs[i].start.total_seconds()
        ref_start = ref_subs[i].start.total_seconds()
        offset = synced_start - ref_start
        offsets.append(offset)

    offsets = np.array(offsets)

    return {
        "median": np.median(offsets),
        "mean": np.mean(offsets),
        "std": np.std(offsets),
        "min": np.min(offsets),
        "max": np.max(offsets),
        "count": len(offsets),
    }


def format_timedelta(seconds: float) -> str:
    """Format seconds as a readable timedelta string."""
    td = timedelta(seconds=abs(seconds))
    sign = "-" if seconds < 0 else "+"
    return f"{sign}{td}"


def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_offset.py <synced_file.srt> <reference_file.srt>")
        print("\nExample:")
        print(
            "  python verify_offset.py output_webrtc.srt test-data/ty/..fi.ref-synced-from-gr.srt"
        )
        sys.exit(1)

    synced_file = sys.argv[1]
    reference_file = sys.argv[2]

    # Check if files exist
    if not Path(synced_file).exists():
        print(f"ERROR: Synced file not found: {synced_file}")
        sys.exit(1)

    if not Path(reference_file).exists():
        print(f"ERROR: Reference file not found: {reference_file}")
        sys.exit(1)

    print("Analyzing offset between:")
    print(f"  Synced:    {synced_file}")
    print(f"  Reference: {reference_file}")
    print()

    stats = calculate_offsets(synced_file, reference_file)

    print("=" * 70)
    print("OFFSET STATISTICS")
    print("=" * 70)
    print(
        f"Median offset: {stats['median']:+.2f}s ({format_timedelta(stats['median'])})"
    )
    print(f"Mean offset:   {stats['mean']:+.2f}s ({format_timedelta(stats['mean'])})")
    print(f"Std deviation: {stats['std']:.2f}s")
    print(f"Min offset:    {stats['min']:+.2f}s ({format_timedelta(stats['min'])})")
    print(f"Max offset:    {stats['max']:+.2f}s ({format_timedelta(stats['max'])})")
    print(f"Subtitles compared: {stats['count']}")
    print("=" * 70)
    print()

    # Check if the offset is acceptable (within ±2 seconds)
    if abs(stats["median"]) <= 2.0 and stats["std"] <= 1.0:
        print("✓ SUCCESS: Synchronization looks good!")
        print("  Median offset is within acceptable range (≤2s)")
    elif abs(stats["median"]) > 50:
        print("✗ FAILURE: Large systematic offset detected!")
        print("  This indicates the algorithm may have locked onto a false positive")
        print("  (e.g., intro music instead of actual speech)")
    else:
        print("⚠ WARNING: Moderate offset detected")
        print("  The synchronization may need refinement")

    print()

    # Return exit code based on success
    if abs(stats["median"]) <= 2.0 and stats["std"] <= 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
