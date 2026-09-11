"""Audio-based piecewise drift correction.

The standard pipeline corrects subtitles with a single affine transform
(``scale * t + offset``), which cannot follow drift that changes rate mid-file.
This module computes *residual* offsets in overlapping windows of the already
extracted 100 Hz binary speech timelines, filters them into a small set of
anchors, and lets the caller apply a smooth monotonic warp between anchors.

Everything here is pure numpy: it runs on the cached speech arrays, so no extra
ffmpeg or VAD pass is needed.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ffsubsync.aligners import FFTAligner

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE_SECONDS = 200.0
DEFAULT_OVERLAP_SECONDS = 100.0
DEFAULT_MAX_RESIDUAL_OFFSET_SECONDS = 15.0
DEFAULT_MIN_SPEECH_SECONDS = 10.0
# A residual jump bigger than this between neighbouring windows is treated as
# an edit difference and placed as a step, not interpolated as drift.
DEFAULT_STEP_THRESHOLD_SECONDS = 2.0


@dataclass(frozen=True)
class WindowOffset:
    """Residual offset measured at the center of one window."""

    center_seconds: float
    offset_seconds: float
    score: float


@dataclass(frozen=True)
class Anchor:
    """A (time, offset) knot of the piecewise warp, in reference time."""

    time: float
    offset: float


def compute_window_offsets(
    ref_speech,
    sub_speech,
    *,
    sample_rate: int = 100,
    window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    max_offset_seconds: float = DEFAULT_MAX_RESIDUAL_OFFSET_SECONDS,
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
) -> list[WindowOffset]:
    """Measure the residual offset between two speech timelines per window.

    Both inputs are binary speech arrays sampled at ``sample_rate`` and are
    expected to be already globally aligned, so the per-window offsets are
    small residuals rather than the full offset.
    """
    ref = np.asarray(ref_speech, dtype=float).ravel()
    sub = np.asarray(sub_speech, dtype=float).ravel()

    window_size_samples = int(window_size_seconds * sample_rate)
    step_size_samples = int((window_size_seconds - overlap_seconds) * sample_rate)
    if step_size_samples <= 0:
        raise ValueError("piecewise overlap must be smaller than the window size")

    max_offset_samples = int(max_offset_seconds * sample_rate)
    min_speech_samples = int(min_speech_seconds * sample_rate)

    usable_len = min(len(ref), len(sub))
    if usable_len < window_size_samples:
        logger.info(
            "too short for piecewise windows (%d < %d samples); no anchors",
            usable_len,
            window_size_samples,
        )
        return []

    last_start_idx = usable_len - window_size_samples
    window_starts = list(range(0, last_start_idx + 1, step_size_samples))
    if window_starts[-1] != last_start_idx:
        window_starts.append(last_start_idx)

    window_offsets: list[WindowOffset] = []
    for start_idx in window_starts:
        end_idx = start_idx + window_size_samples
        ref_window = ref[start_idx:end_idx]
        sub_window = sub[start_idx:end_idx]

        # A window with almost no speech on either side cannot produce a
        # meaningful correlation peak.
        if (
            ref_window.sum() < min_speech_samples
            or sub_window.sum() < min_speech_samples
        ):
            continue

        try:
            aligner = FFTAligner(max_offset_samples=max_offset_samples)
            aligner.fit(ref_window, sub_window, get_score=True)
        except Exception as e:
            logger.warning("piecewise window at sample %d failed: %s", start_idx, e)
            continue

        if aligner.best_offset_ is None or aligner.best_score_ is None:
            continue

        center_samples = start_idx + window_size_samples / 2.0
        window_offsets.append(
            WindowOffset(
                center_seconds=center_samples / sample_rate,
                offset_seconds=aligner.best_offset_ / float(sample_rate),
                score=float(aligner.best_score_),
            )
        )

    return window_offsets


def _median_filter(values: Sequence[float], size: int) -> list[float]:
    half = size // 2
    filtered = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        filtered.append(float(np.median(values[lo:hi])))
    return filtered


def build_anchors(
    window_offsets: Sequence[WindowOffset],
    *,
    score_percentile: float = 25.0,
    max_outlier_seconds: float = 5.0,
    monotonicity_margin_seconds: float = 0.5,
) -> list[Anchor]:
    """Turn raw per-window offsets into a trustworthy, monotonic anchor set.

    Returns an empty list when fewer than two anchors survive, which the caller
    should read as "keep the global-only result".
    """
    if len(window_offsets) < 2:
        return []

    # Raw FFT scores are unnormalized and routinely go negative, so any
    # threshold relative to the mean or median breaks as soon as scores
    # straddle zero. Rank instead: drop the weakest quarter of windows and let
    # the outlier and monotonicity checks below do the real filtering.
    scores = [w.score for w in window_offsets]
    if len(window_offsets) >= 6:
        threshold = float(np.percentile(scores, score_percentile))
        candidates = [w for w in window_offsets if w.score >= threshold]
    else:
        candidates = list(window_offsets)
    if len(candidates) < 2:
        return []

    offsets = [w.offset_seconds for w in candidates]
    filter_size = 5 if len(candidates) >= 7 else 3
    smoothed = _median_filter(offsets, filter_size)
    candidates = [
        w
        for w, ref_offset in zip(candidates, smoothed, strict=True)
        if abs(w.offset_seconds - ref_offset) <= max_outlier_seconds
    ]
    if len(candidates) < 2:
        return []

    # The warp t -> t + offset(t) must stay increasing, otherwise cues would be
    # reordered. Where a pair violates that, drop the lower-scoring window.
    kept: list[WindowOffset] = [candidates[0]]
    for window in candidates[1:]:
        previous = kept[-1]
        dt = window.center_seconds - previous.center_seconds
        d_offset = window.offset_seconds - previous.offset_seconds
        if dt + d_offset <= monotonicity_margin_seconds:
            if window.score > previous.score:
                kept[-1] = window
            continue
        kept.append(window)

    if len(kept) < 2:
        return []

    return [Anchor(time=w.center_seconds, offset=w.offset_seconds) for w in kept]


def _longest_silence(
    lo: float, hi: float, cue_spans: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """Return the longest stretch of ``[lo, hi]`` that no cue covers."""
    best = (lo, lo)
    cursor = lo
    for start, end in sorted(cue_spans):
        if end <= lo or start >= hi:
            continue
        if start > cursor and start - cursor > best[1] - best[0]:
            best = (cursor, start)
        cursor = max(cursor, end)
    if hi > cursor and hi - cursor > best[1] - best[0]:
        best = (cursor, hi)
    return best


def snap_steps_to_gaps(
    anchors: Sequence[Anchor],
    cue_spans: Sequence[tuple[float, float]],
    *,
    step_threshold_seconds: float = DEFAULT_STEP_THRESHOLD_SECONDS,
    monotonicity_margin_seconds: float = 0.5,
) -> list[Anchor]:
    """Turn large jumps between adjacent anchors into steps at a cue-free gap.

    A residual that jumps by several seconds between two neighbouring windows
    is an edit difference (a scene cut or trimmed shot), not gradual drift.
    Interpolating linearly across it would push every cue between the two
    anchors partway, so none of them lands right. Instead, find the longest
    silence in the subtitle timeline between the two anchors and move the
    whole jump there: cues before the silence keep the earlier offset, cues
    after it take the later one.

    ``cue_spans`` are ``(start, end)`` pairs in the same timeline as the
    anchors (reference time, after the global shift). A pair is left as a
    ramp when the silence is too short to absorb a negative jump, because the
    warp would then reorder cues.
    """
    if len(anchors) < 2:
        return list(anchors)

    refined: list[Anchor] = [anchors[0]]
    for a, b in pairwise(anchors):
        jump = b.offset - a.offset
        if abs(jump) > step_threshold_seconds:
            gap_start, gap_end = _longest_silence(a.time, b.time, cue_spans)
            if gap_end - gap_start + jump <= monotonicity_margin_seconds:
                logger.info(
                    "piecewise: edit step %+.3fs between %.1fs and %.1fs does "
                    "not fit the longest silence (%.1f-%.1fs); keeping the ramp",
                    jump,
                    a.time,
                    b.time,
                    gap_start,
                    gap_end,
                )
            else:
                logger.info(
                    "piecewise: edit step %+.3fs placed in the silence %.1f-%.1fs",
                    jump,
                    gap_start,
                    gap_end,
                )
                if gap_start > refined[-1].time:
                    refined.append(Anchor(time=gap_start, offset=a.offset))
                if gap_end < b.time:
                    refined.append(Anchor(time=gap_end, offset=b.offset))
        refined.append(b)
    return refined
