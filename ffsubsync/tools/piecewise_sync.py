#!/usr/bin/env python3
"""
Piecewise Subtitle Synchronization Tool

This tool handles difficult sync cases where subtitles drift mid-file due to:
- Different video edits (commercial breaks removed differently)
- Framerate mismatches that vary throughout the file
- Subtitles from different source releases

Algorithm:
1. Ratio-based time warping: Maps source subtitle timeline proportionally to reference timeline
2. Local offset correction: Finds optimal offset per time window using median filtering
3. Smooth interpolation: Blends corrections to avoid jarring jumps

Usage:
    python -m ffsubsync.tools.piecewise_sync reference.srt input.srt output.srt
    piecewise-sync reference.srt input.srt output.srt
"""

import argparse
import re
from collections import defaultdict
from typing import Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


def parse_srt(filepath: str) -> tuple[list[dict], str]:
    """
    Parse SRT file while preserving original encoding.

    Returns:
        Tuple of (list of subtitle dicts, detected encoding)
    """
    # Detect encoding
    encoding = "utf-8"
    if HAS_CHARDET:
        with open(filepath, "rb") as f:
            raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
        content = raw.decode(encoding, errors="replace")
    else:
        # Try UTF-8 first, fall back to latin-1
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, encoding="latin-1") as f:
                content = f.read()
            encoding = "latin-1"

    blocks = re.split(r"\n\n+", content.strip())
    subs = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            try:
                idx = int(lines[0])
            except ValueError:
                continue

            timing_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})",
                lines[1],
            )
            if timing_match:
                m = timing_match.groups()
                start_ms = (
                    int(m[0]) * 3600000
                    + int(m[1]) * 60000
                    + int(m[2]) * 1000
                    + int(m[3])
                )
                end_ms = (
                    int(m[4]) * 3600000
                    + int(m[5]) * 60000
                    + int(m[6]) * 1000
                    + int(m[7])
                )
                text = "\n".join(lines[2:]) if len(lines) > 2 else ""
                subs.append(
                    {"idx": idx, "start": start_ms, "end": end_ms, "text": text}
                )

    return subs, encoding


def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)"""
    if ms < 0:
        ms = 0
    hours = ms // 3600000
    mins = (ms % 3600000) // 60000
    secs = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def write_srt(subs: list[dict], filepath: str, encoding: str = "utf-8-sig") -> None:
    """
    Write subtitles to SRT file.

    Uses UTF-8 with BOM by default for maximum compatibility with media players.
    """
    with open(filepath, "w", encoding=encoding) as f:
        for i, sub in enumerate(subs, 1):
            f.write(f"{i}\n")
            f.write(
                f"{ms_to_srt_time(sub['start'])} --> {ms_to_srt_time(sub['end'])}\n"
            )
            f.write(f"{sub['text']}\n\n")


def find_matching_reference(
    input_sub: dict, ref_subs: list[dict], search_window_ms: int = 60000
) -> tuple[Optional[dict], int]:
    """
    Find the reference subtitle that best matches the input subtitle.

    Matching is based on duration similarity and temporal proximity.

    Returns:
        Tuple of (matching reference subtitle, offset in ms)
    """
    input_start = input_sub["start"]
    input_dur = input_sub["end"] - input_sub["start"]

    candidates = []
    for ref_sub in ref_subs:
        time_diff = abs(ref_sub["start"] - input_start)
        if time_diff > search_window_ms:
            continue

        ref_dur = ref_sub["end"] - ref_sub["start"]
        dur_diff = abs(ref_dur - input_dur)

        # Score: prefer similar duration and close timing
        score = 1000 - dur_diff - time_diff / 10
        offset = ref_sub["start"] - input_sub["start"]
        candidates.append((ref_sub, score, offset))

    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0], candidates[0][2]
    return None, 0


def calculate_local_corrections(
    warped_subs: list[dict],
    ref_subs: list[dict],
    window_size_ms: int = 60000,
    min_correction_ms: int = 300,
) -> dict[int, int]:
    """
    Calculate local offset corrections per time window.

    Uses median filtering to reject outliers and find robust local offsets.

    Returns:
        Dict mapping window start time (ms) to correction offset (ms)
    """
    # Build offset map for each subtitle
    offsets_by_time = []
    for sub in warped_subs:
        match, offset = find_matching_reference(sub, ref_subs)
        if match:
            offsets_by_time.append({"time": sub["start"], "offset": offset})

    # Group by time window and calculate median
    window_offsets = defaultdict(list)
    for item in offsets_by_time:
        window = item["time"] // window_size_ms
        window_offsets[window].append(item["offset"])

    corrections = {}
    for window in sorted(window_offsets.keys()):
        offsets = window_offsets[window]
        if len(offsets) >= 2:
            if HAS_NUMPY:
                median_offset = int(np.median(offsets))
            else:
                sorted_offsets = sorted(offsets)
                mid = len(sorted_offsets) // 2
                median_offset = sorted_offsets[mid]

            if abs(median_offset) > min_correction_ms:
                corrections[window * window_size_ms] = median_offset

    return corrections


def get_interpolated_correction(
    time_ms: int, corrections: dict[int, int], window_size_ms: int
) -> int:
    """
    Get smoothly interpolated correction for a given time.

    Interpolates between neighboring corrections to avoid jarring jumps.
    """
    window = (time_ms // window_size_ms) * window_size_ms

    if window in corrections:
        return corrections[window]

    # Interpolate from neighbors
    prev_window = window - window_size_ms
    next_window = window + window_size_ms

    if prev_window in corrections and next_window in corrections:
        ratio = (time_ms - prev_window) / (2 * window_size_ms)
        return int(
            corrections[prev_window] * (1 - ratio) + corrections[next_window] * ratio
        )
    elif prev_window in corrections:
        ratio = min(1.0, (time_ms - prev_window) / window_size_ms)
        return int(corrections[prev_window] * (1 - ratio * 0.5))
    elif next_window in corrections:
        ratio = min(1.0, (next_window - time_ms) / window_size_ms)
        return int(corrections[next_window] * (1 - ratio * 0.5))

    return 0


def piecewise_sync(
    ref_subs: list[dict], input_subs: list[dict], window_size_ms: int = 60000
) -> list[dict]:
    """
    Perform piecewise synchronization of input subtitles to reference.

    Algorithm:
    1. Ratio-based time warping: Map input timeline to reference timeline
    2. Local offset correction: Find and apply per-window corrections
    3. Smooth interpolation: Blend corrections between windows

    Args:
        ref_subs: Reference subtitles (correctly timed)
        input_subs: Input subtitles to sync
        window_size_ms: Size of correction windows in milliseconds

    Returns:
        List of synchronized subtitle dicts
    """
    if not input_subs or not ref_subs:
        return input_subs

    # Step 1: Ratio-based time warping
    input_total_duration = input_subs[-1]["end"] - input_subs[0]["start"]
    ref_total_duration = ref_subs[-1]["end"] - ref_subs[0]["start"]

    if input_total_duration <= 0:
        return input_subs

    input_start_offset = input_subs[0]["start"]
    ref_start_offset = ref_subs[0]["start"]
    ref_total_duration / input_total_duration

    warped_subs = []
    for sub in input_subs:
        rel_pos = (sub["start"] - input_start_offset) / input_total_duration
        new_start = ref_start_offset + rel_pos * ref_total_duration
        duration = sub["end"] - sub["start"]
        new_end = new_start + duration

        warped_subs.append(
            {
                "idx": sub["idx"],
                "start": max(0, int(new_start)),
                "end": max(100, int(new_end)),
                "text": sub["text"],
            }
        )

    # Step 2: Calculate local corrections
    corrections = calculate_local_corrections(warped_subs, ref_subs, window_size_ms)

    # Step 3: Apply corrections with smooth interpolation
    final_subs = []
    for sub in warped_subs:
        correction = get_interpolated_correction(
            sub["start"], corrections, window_size_ms
        )
        new_start = sub["start"] + correction
        new_end = sub["end"] + correction
        final_subs.append(
            {
                "idx": sub["idx"],
                "start": max(0, new_start),
                "end": max(new_start + 100, new_end),
                "text": sub["text"],
            }
        )

    return final_subs


def verify_sync(
    synced_subs: list[dict], ref_subs: list[dict], threshold_ms: int = 1000
) -> dict:
    """
    Verify sync quality by comparing against reference.

    Returns:
        Dict with verification metrics
    """

    def find_closest_gap(subs: list[dict], target_ms: int) -> int:
        best_gap = float("inf")
        for sub in subs:
            if sub["start"] <= target_ms <= sub["end"]:
                return 0
            gap = min(abs(sub["start"] - target_ms), abs(sub["end"] - target_ms))
            if gap < best_gap:
                best_gap = gap
        return int(best_gap) if best_gap != float("inf") else -1

    gaps = [find_closest_gap(synced_subs, s["start"]) for s in ref_subs]
    valid_gaps = [g for g in gaps if g >= 0]

    if not valid_gaps:
        return {
            "within_threshold": 0,
            "total": len(ref_subs),
            "max_gap": -1,
            "pct": 0.0,
        }

    within_threshold = sum(1 for g in valid_gaps if g <= threshold_ms)
    max_gap = max(valid_gaps)

    return {
        "within_threshold": within_threshold,
        "total": len(ref_subs),
        "max_gap": max_gap,
        "pct": 100 * within_threshold / len(ref_subs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Piecewise subtitle synchronization for difficult drift cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s french.srt finnish.srt synced_finnish.srt
  %(prog)s --window 30000 reference.srt input.srt output.srt
  %(prog)s --verify reference.srt input.srt output.srt

This tool handles cases where standard ffsubsync fails due to:
- Non-linear drift (subtitles from different video edits)
- Mid-file timing changes (commercial breaks removed differently)
- Different subtitle segmentation between translations
        """,
    )
    parser.add_argument("reference", help="Reference subtitle file (correctly timed)")
    parser.add_argument("input", help="Input subtitle file to sync")
    parser.add_argument("output", help="Output synchronized subtitle file")
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=60000,
        help="Correction window size in milliseconds (default: 60000)",
    )
    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Print verification metrics after sync",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=1000,
        help="Verification threshold in milliseconds (default: 1000)",
    )

    args = parser.parse_args()

    # Parse input files
    print(f"Loading reference: {args.reference}")
    ref_subs, _ = parse_srt(args.reference)

    print(f"Loading input: {args.input}")
    input_subs, input_encoding = parse_srt(args.input)

    print(f"Reference: {len(ref_subs)} subtitles")
    print(f"Input: {len(input_subs)} subtitles (encoding: {input_encoding})")

    # Perform sync
    print(f"Syncing with {args.window}ms windows...")
    synced_subs = piecewise_sync(ref_subs, input_subs, args.window)

    # Write output with UTF-8 BOM for compatibility
    write_srt(synced_subs, args.output)
    print(f"Written: {args.output}")

    # Verify if requested
    if args.verify:
        metrics = verify_sync(synced_subs, ref_subs, args.threshold)
        print(f"\nVerification (threshold: {args.threshold}ms):")
        print(
            f"  Within threshold: {metrics['within_threshold']}/{metrics['total']} ({metrics['pct']:.1f}%)"
        )
        print(f"  Max gap: {metrics['max_gap']}ms ({metrics['max_gap'] / 1000:.1f}s)")


if __name__ == "__main__":
    main()
