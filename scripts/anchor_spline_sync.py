#!/usr/bin/env python3
"""
Piecewise (spline) subtitle synchronizer using auto-detected anchors.

Goal: map source subtitle timeline -> reference (EN) timeline using a
monotonic piecewise linear warp derived from multiple anchors spread across
the episode. Anchors are detected by:

1) Computing a coarse global offset via ffsubsync's FFT alignment of
   subtitle-derived speech sequences (no video needed).
2) Dividing the EN timeline into segments and, for each segment center, picking
   the nearest EN line and the nearest FI line around the coarse-offset-predicted
   time, within a small window. These pairs become anchors.
3) Optionally, users can add manual anchors as safeguards.

This approach is robust when subtitles differ in segmentation or contain extra
SDH/recap/credits, while avoiding overfitting to two points.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import srt

from ffsubsync.aligners import FFTAligner
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.speech_transformers import make_subtitle_speech_pipeline


def _parse_time(value: str) -> float:
    v = value.strip()
    try:
        return float(v)
    except ValueError:
        pass
    parts = v.split(":")
    parts = [p.strip() for p in parts]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return 60.0 * float(m) + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return 3600.0 * float(h) + 60.0 * float(m) + float(s)
    raise argparse.ArgumentTypeError(f"Invalid time format: {value}")


def _load_srt(path: str) -> list[srt.Subtitle]:
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8", "replace")
    return [x for x in srt.parse(txt) if x.content.strip()]


def _starts(subs: Sequence[srt.Subtitle]) -> list[float]:
    return [s.start.total_seconds() for s in subs]


def _bisect_left(a: Sequence[float], x: float) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _coarse_offset_seconds(
    ref_en_path: str, src_fi_path: str, max_offset_s: float = 600.0
) -> float:
    """Compute coarse offset to apply to FI to align to EN (seconds)."""
    en_pipe = make_subtitle_speech_pipeline(start_seconds=0)
    fi_pipe = make_subtitle_speech_pipeline(start_seconds=0)
    en = en_pipe.fit_transform(ref_en_path)
    fi = fi_pipe.fit_transform(src_fi_path)
    aligner = FFTAligner(max_offset_samples=int(max_offset_s * SAMPLE_RATE))
    _, offset_samples = aligner.fit_transform(en, fi, get_score=True)
    return float(offset_samples) / float(SAMPLE_RATE)


@dataclass
class Anchor:
    src: float  # seconds in source timeline
    ref: float  # seconds in reference timeline


def _auto_anchors(
    en_subs: Sequence[srt.Subtitle],
    fi_subs: Sequence[srt.Subtitle],
    coarse_offset: float,
    segments: int,
    window: float,
    min_anchors: int,
) -> list[Anchor]:
    en_starts = _starts(en_subs)
    fi_starts = _starts(fi_subs)
    if not en_starts or not fi_starts:
        return []
    t0, t1 = en_starts[0], en_starts[-1]
    anchors: list[Anchor] = []
    for k in range(segments):
        # Center time for this segment in EN timeline
        u = (k + 0.5) / segments
        te = t0 + u * (t1 - t0)
        j = _bisect_left(en_starts, te)
        if j == len(en_starts):
            j -= 1
        te = en_starts[j]
        # Predict FI time via coarse offset
        tf_pred = te - coarse_offset
        i = _bisect_left(fi_starts, tf_pred)
        candidates: list[tuple[float, int]] = []
        if i < len(fi_starts):
            candidates.append((abs(fi_starts[i] - tf_pred), i))
        if i > 0:
            candidates.append((abs(fi_starts[i - 1] - tf_pred), i - 1))
        if not candidates:
            continue
        d, idx = min(candidates, key=lambda x: x[0])
        if d <= window:
            anchors.append(Anchor(src=fi_starts[idx], ref=te))
    # If too few anchors, widen window progressively
    widen = window
    while len(anchors) < min_anchors and widen <= 5.0:
        widen *= 1.5
        anchors = _auto_anchors(
            en_subs, fi_subs, coarse_offset, segments, widen, min_anchors
        )
        if len(anchors) >= min_anchors:
            break
    # Deduplicate by src time proximity
    anchors.sort(key=lambda a: a.src)
    dedup: list[Anchor] = []
    last_src = -1e9
    for a in anchors:
        if a.src - last_src > 0.5:  # 500 ms separation
            dedup.append(a)
            last_src = a.src
    return dedup


def _build_piecewise_map(anchors: Sequence[Anchor]):
    """
    Returns a monotonic piecewise-linear mapping f(t)=a*t+b defined by anchors,
    supporting linear extrapolation at ends.
    """
    if len(anchors) < 2:
        raise ValueError("Need at least 2 anchors for piecewise mapping")
    anchors = sorted(anchors, key=lambda a: a.src)
    segs: list[tuple[float, float, float]] = []  # (src_left, a, b)
    for i in range(len(anchors) - 1):
        x1, y1 = anchors[i].src, anchors[i].ref
        x2, y2 = anchors[i + 1].src, anchors[i + 1].ref
        if math.isclose(x2, x1):
            a = 1.0
            b = y1 - x1
        else:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
        segs.append((x1, a, b))

    def f(t: float) -> float:
        if t <= anchors[0].src:
            a, b = segs[0][1], segs[0][2]
            return a * t + b
        if t >= anchors[-1].src:
            a, b = segs[-1][1], segs[-1][2]
            return a * t + b
        # find segment by linear scan (anchors are modest)
        for i in range(len(segs)):
            left = segs[i][0]
            right = anchors[i + 1].src
            if left <= t <= right:
                a, b = segs[i][1], segs[i][2]
                return a * t + b
        # fallback
        a, b = segs[-1][1], segs[-1][2]
        return a * t + b

    return f


def _write_srt(subs: Sequence[srt.Subtitle], path: str, map_fn) -> None:
    out = []
    for s in subs:
        ns = srt.Subtitle(
            index=s.index,
            start=s.start,
            end=s.end,
            content=s.content,
            proprietary=s.proprietary,
        )
        ns.start = srt.timedelta(seconds=max(0.0, map_fn(s.start.total_seconds())))
        ns.end = srt.timedelta(
            seconds=max(ns.start.total_seconds(), map_fn(s.end.total_seconds()))
        )
        out.append(ns)
    text = srt.compose(out)
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Piecewise auto-anchored subtitle sync (FI -> EN)"
    )
    ap.add_argument(
        "-i", "--srtin", required=True, help="Input source subtitles (e.g., Finnish)"
    )
    ap.add_argument(
        "-r", "--reference", required=True, help="Reference subtitles (e.g., English)"
    )
    ap.add_argument("-o", "--srtout", required=True, help="Output subtitle path")
    ap.add_argument(
        "--segments",
        type=int,
        default=18,
        help="Number of segments across EN timeline (default=18)",
    )
    ap.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Max seconds to accept FI/EN anchor pair (default=1.0)",
    )
    ap.add_argument(
        "--min-anchors", type=int, default=8, help="Minimum anchors to keep (default=8)"
    )
    ap.add_argument(
        "--max-offset",
        type=float,
        default=600.0,
        help="Max abs offset for coarse search (s)",
    )
    ap.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="Manual anchor src=..,ref=.. (repeatable)",
    )
    args = ap.parse_args()

    en = _load_srt(args.reference)
    fi = _load_srt(args.srtin)
    coarse = _coarse_offset_seconds(
        args.reference, args.srtin, max_offset_s=args.max_offset
    )
    print(f"Coarse offset (apply to FI -> EN): {coarse:.3f} s")
    anchors = _auto_anchors(
        en, fi, coarse, args.segments, args.window, args.min_anchors
    )

    # add manual anchors
    for spec in args.anchor:
        parts = dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)
        src = _parse_time(parts["src"]) if "src" in parts else None
        ref = _parse_time(parts["ref"]) if "ref" in parts else None
        if src is None or ref is None:
            raise ValueError(f"Invalid --anchor spec: {spec}")
        anchors.append(Anchor(src=src, ref=ref))

    anchors.sort(key=lambda a: a.src)
    # Print summary
    print(f"Anchors selected: {len(anchors)}")
    for a in anchors[:10]:
        print(f"  src={a.src:8.3f} -> ref={a.ref:8.3f}")
    if len(anchors) > 10:
        print("  …")

    if len(anchors) < 2:
        raise SystemExit(
            "Not enough anchors detected. Try increasing --window or --segments."
        )

    f = _build_piecewise_map(anchors)
    _write_srt(fi, args.srtout, f)
    print(f"Wrote: {args.srtout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
