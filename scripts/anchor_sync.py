#!/usr/bin/env python3
"""
Two-point anchor subtitle synchronizer using ffsubsync internals.

Given two anchor mappings (source time -> reference/correct time), this tool
computes a linear warp t' = a * t + b and applies it to the entire subtitle
file (scale, then shift). This mirrors the manual "fix first and last line"
workflow.

Examples
  # First line should start at 00:00:08.500 and last line should end at 00:58:31.900
  python scripts/anchor_sync.py \
    -i "input.srt" -o "output.synced.srt" \
    --t1-src 00:00:08.000 --t1-ref 00:00:08.500 \
    --t2-src 00:58:35.123 --t2-ref 00:58:31.900

Times may be given as HH:MM:SS[.ms], MM:SS[.ms], or seconds as a float.
"""

from __future__ import annotations

import argparse

from ffsubsync.sklearn_shim import Pipeline
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleScaler, SubtitleShifter


def _parse_time(value: str) -> float:
    """Parse time in HH:MM:SS[.ms], MM:SS[.ms], or float seconds.

    Returns seconds as float.
    """
    v = value.strip()
    # float seconds
    try:
        return float(v)
    except ValueError:
        pass
    # tokenized clock time
    parts = v.split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(f"Invalid time format: {value}")
    parts = [p.strip() for p in parts]
    # seconds (with optional ms)
    if len(parts) == 1:
        s = float(parts[0])
        return s
    # mm:ss[.ms]
    if len(parts) == 2:
        m, s = parts
        return 60.0 * float(m) + float(s)
    # hh:mm:ss[.ms]
    h, m, s = parts
    return 3600.0 * float(h) + 60.0 * float(m) + float(s)


def _compute_affine(
    t1_src: float, t1_ref: float, t2_src: float, t2_ref: float
) -> tuple[float, float]:
    """Return (scale a, offset b) such that t' = a * t + b.

    Raises if anchors are degenerate.
    """
    denom = t2_src - t1_src
    if abs(denom) < 1e-9:
        raise ValueError("Anchor times are identical on source; cannot compute scale.")
    a = (t2_ref - t1_ref) / denom
    b = t1_ref - a * t1_src
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply two-point anchor sync to an SRT.")
    ap.add_argument(
        "-i", "--srtin", required=True, help="Input subtitle file (SRT/ASS/SSA/SUB)."
    )
    ap.add_argument(
        "-o", "--srtout", required=True, help="Output subtitle file (SRT by extension)."
    )
    ap.add_argument(
        "--t1-src",
        required=True,
        type=_parse_time,
        help="First anchor time in source (seconds or clock).",
    )
    ap.add_argument(
        "--t1-ref",
        required=True,
        type=_parse_time,
        help="First anchor time in reference/correct timeline.",
    )
    ap.add_argument(
        "--t2-src",
        required=True,
        type=_parse_time,
        help="Second anchor time in source (seconds or clock).",
    )
    ap.add_argument(
        "--t2-ref",
        required=True,
        type=_parse_time,
        help="Second anchor time in reference/correct timeline.",
    )
    ap.add_argument(
        "--output-encoding",
        default="same",
        help='Output encoding (default="same" to preserve input encoding).',
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the computed transform and exit."
    )
    args = ap.parse_args()

    a, b = _compute_affine(args.t1_src, args.t1_ref, args.t2_src, args.t2_ref)
    # Informative print; intentionally simple stdout
    print(f"Computed scale a={a:.9f}, offset b={b:.3f} seconds")

    if args.dry_run:
        return 0

    # Parse input subtitles with encoding inference
    parser = make_subtitle_parser(
        fmt=args.srtin.split(".")[-1].lower(), encoding="infer"
    )
    subs = parser.fit(args.srtin).transform()

    # Apply scale then shift
    pipe = Pipeline([("scale", SubtitleScaler(a)), ("shift", SubtitleShifter(b))])
    out = pipe.fit_transform(subs)
    out.set_encoding(args.output_encoding)
    out.write_file(args.srtout)
    print(f"Wrote: {args.srtout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
