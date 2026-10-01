"""Cut-aware alignment of a subtitle against a subtitle reference in another cut.

A subtitle made for a shorter cut of a film needs one offset per scene when the
video is a longer cut: every added scene pushes the following cues later. The
offset function is piecewise constant with many forward jumps (20 or more in one
episode), which neither a global fit nor :mod:`ffsubsync.split_aligner` against
a speech timeline finds reliably.

The reference here is a *text* subtitle track of the video (usually an embedded
stream in another language), reduced to its cue timings. Two things make the
problem tractable:

* **Cue starts, not speech overlap.** Translations of the same programme are cut
  into cues at the same moments, so the score of a cue at an offset is mostly a
  narrow kernel around the reference cue starts. Overlap with the reference
  dialogue is only a tie-breaker. Overlap alone has many false optima.
* **Forward jumps.** The offset may stay or jump forward, at a fixed price per
  jump. A small step back (``max_back_step``) is allowed, because the cues of a
  recap or a re-timed line can sit slightly earlier than the following scene.

The dynamic program maximizes::

    sum_i score(cue_i @ offset_i)  -  jump_penalty * (number of offset changes)

``anchors`` pin single cues to an offset. :mod:`ffsubsync.ai_judge` produces
them by matching cue texts across the two languages, for the stretches where
the timing score alone picks the wrong scene.

This module does no I/O and does not look at cue text, except for
:func:`is_dialogue_text`.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ffsubsync.constants import SAMPLE_RATE

logger: logging.Logger = logging.getLogger(__name__)

# Offset grid in samples (40 ms at 100 Hz): finer than a video frame.
OFFSET_STEP_SAMPLES: int = 4
# Half-width in samples of the triangular kernel around a reference cue start.
START_KERNEL_SAMPLES: int = 40
START_WEIGHT: float = 2.0
OVERLAP_WEIGHT: float = 0.5
# Price of one offset change, in score units (a perfect cue scores about 2.5).
DEFAULT_JUMP_PENALTY: float = 3.0
DEFAULT_MAX_BACK_STEP_SECONDS: float = 1.0
DEFAULT_OFFSET_MARGIN_SECONDS: float = 60.0
# An anchored cue may only take offsets this close to its anchor. An anchor says
# which scene a cue belongs to; the cue starts still decide the exact offset,
# because a translated cue need not begin on the same word as its counterpart.
ANCHOR_TOLERANCE_SECONDS: float = 2.5
# A cue "agrees" with the reference when a reference cue starts this close.
AGREEMENT_SECONDS: float = 0.5
_FORBIDDEN: float = -1e9

Times = Sequence[tuple[float, float]]

_HAS_LOWERCASE = re.compile(r"[a-zà-öø-ÿ]")
_BRACKETED = re.compile(r"^\s*[\[(].*[\])]\s*$", re.DOTALL)
_MIN_SOUND_DESCRIPTION_LETTERS = 4


def is_dialogue_text(text: str) -> bool:
    """False for cues that only describe sound (``SIREN WAILS``, ``[music]``).

    Hearing-impaired tracks carry these between the spoken lines. A translation
    has no cue for them, so they must not attract cue starts.
    """
    stripped = text.strip()
    if not stripped or _BRACKETED.match(stripped):
        return False
    if _HAS_LOWERCASE.search(stripped):
        return True
    # No lowercase: a sound description, unless it is too short to be one
    # ("OK?", "27, 28!").
    return sum(ch.isalpha() for ch in stripped) < _MIN_SOUND_DESCRIPTION_LETTERS


@dataclass(frozen=True)
class CutSegment:
    """A run of consecutive cues that share one offset (0-based, inclusive)."""

    first: int
    last: int
    offset_seconds: float
    # Share of the cues with a reference cue start within AGREEMENT_SECONDS.
    agreement: float


@dataclass(frozen=True)
class CutAlignment:
    offsets: list[float]
    segments: list[CutSegment]
    objective: float
    # Timing score per cue net of the jump penalties, anchors not counted. About
    # 2.5 at best; a subtitle of the same programme scores well above 1, one that
    # only fits by chance below (see ssync's AI_MIN_SCORE_PER_CUE).
    score_per_cue: float = 0.0


@dataclass(frozen=True)
class CutAssessment:
    within_half_second: float
    within_one_second: float
    # (start_seconds, end_seconds, cue_count) of reference dialogue no cue covers.
    uncovered: list[tuple[float, float, int]]


def default_offset_range(
    sub_times: Times,
    ref_times: Times,
    margin_seconds: float = DEFAULT_OFFSET_MARGIN_SECONDS,
) -> tuple[float, float]:
    """Offsets worth searching: a margin back, and the length difference forward."""
    sub_end = max((end for _, end in sub_times), default=0.0)
    ref_end = max((end for _, end in ref_times), default=0.0)
    return -margin_seconds, max(margin_seconds, ref_end - sub_end + margin_seconds)


def _start_distances(starts: np.ndarray, ref_starts: np.ndarray) -> np.ndarray:
    """Distance in seconds from each start to the nearest reference start."""
    if len(ref_starts) == 0:
        return np.full(len(starts), np.inf)
    pos = np.searchsorted(ref_starts, starts)
    left = ref_starts[np.clip(pos - 1, 0, len(ref_starts) - 1)]
    right = ref_starts[np.clip(pos, 0, len(ref_starts) - 1)]
    return np.minimum(np.abs(starts - left), np.abs(starts - right))


def _segments(
    offsets: Sequence[float], sub_times: Times, ref_times: Times
) -> list[CutSegment]:
    ref_starts = np.sort(np.array([start for start, _ in ref_times], dtype=float))
    shifted = np.array(
        [start + off for (start, _), off in zip(sub_times, offsets, strict=True)]
    )
    agrees = _start_distances(shifted, ref_starts) <= AGREEMENT_SECONDS
    segments: list[CutSegment] = []
    first = 0
    for i in range(1, len(offsets) + 1):
        if i == len(offsets) or offsets[i] != offsets[first]:
            segments.append(
                CutSegment(
                    first=first,
                    last=i - 1,
                    offset_seconds=float(offsets[first]),
                    agreement=float(agrees[first:i].mean()),
                )
            )
            first = i
    return segments


def compute_cut_offsets(
    sub_times: Times,
    ref_times: Times,
    *,
    min_offset: float | None = None,
    max_offset: float | None = None,
    jump_penalty: float = DEFAULT_JUMP_PENALTY,
    max_back_step: float = DEFAULT_MAX_BACK_STEP_SECONDS,
    anchors: Mapping[int, float] | None = None,
) -> CutAlignment:
    """One offset per cue, piecewise constant, jumping forward between scenes.

    Args:
        sub_times: ``(start, end)`` seconds of every cue to move, in file order.
        ref_times: ``(start, end)`` seconds of the reference *dialogue* cues
            (see :func:`is_dialogue_text`).
        min_offset, max_offset: Search range in seconds. Defaults to
            :func:`default_offset_range`.
        jump_penalty: Score a change of offset must gain.
        max_back_step: Largest step back in seconds between consecutive cues.
        anchors: ``{cue index: offset seconds}`` for cues whose place is known.

    Returns:
        A :class:`CutAlignment`. With no cues the offsets are empty.
    """
    if not sub_times:
        return CutAlignment(offsets=[], segments=[], objective=0.0)
    lo_default, hi_default = default_offset_range(sub_times, ref_times)
    lo = lo_default if min_offset is None else min_offset
    hi = hi_default if max_offset is None else max_offset

    grid = np.arange(
        round(lo * SAMPLE_RATE / OFFSET_STEP_SAMPLES),
        round(hi * SAMPLE_RATE / OFFSET_STEP_SAMPLES) + 1,
    )
    offsets_samples = grid * OFFSET_STEP_SAMPLES
    n_offsets = len(offsets_samples)

    sub_end = max(end for _, end in sub_times)
    ref_end = max((end for _, end in ref_times), default=0.0)
    n_samples = int((max(sub_end + hi, ref_end) + 2.0) * SAMPLE_RATE)

    # +1 inside reference dialogue, -1 outside, as a prefix sum for O(1) overlap.
    signal = np.full(n_samples, -1.0)
    start_kernel = np.zeros(n_samples + 1)
    ramp = 1.0 - np.abs(np.arange(-START_KERNEL_SAMPLES, START_KERNEL_SAMPLES + 1)) / (
        START_KERNEL_SAMPLES
    )
    for start, end in ref_times:
        a = max(round(start * SAMPLE_RATE), 0)
        b = min(round(end * SAMPLE_RATE), n_samples)
        signal[a:b] = 1.0
        k0 = a - START_KERNEL_SAMPLES
        k_lo, k_hi = max(k0, 0), min(k0 + len(ramp), n_samples + 1)
        if k_lo < k_hi:
            start_kernel[k_lo:k_hi] = np.maximum(
                start_kernel[k_lo:k_hi], ramp[k_lo - k0 : k_hi - k0]
            )
    prefix = np.concatenate([[0.0], np.cumsum(signal)])

    def timing_score(i: int) -> np.ndarray:
        start, end = sub_times[i]
        a = np.clip(round(start * SAMPLE_RATE) + offsets_samples, 0, n_samples)
        b = np.clip(round(end * SAMPLE_RATE) + offsets_samples, 0, n_samples)
        duration = np.maximum(b - a, SAMPLE_RATE // 2)
        return START_WEIGHT * start_kernel[a] + OVERLAP_WEIGHT * (
            (prefix[b] - prefix[a]) / duration
        )

    def emission(i: int) -> np.ndarray:
        score = timing_score(i)
        if anchors is not None and i in anchors:
            far = (
                np.abs(offsets_samples / SAMPLE_RATE - anchors[i])
                > ANCHOR_TOLERANCE_SECONDS
            )
            score = np.where(far, _FORBIDDEN, score)
        return score

    back_steps = round(max_back_step * SAMPLE_RATE / OFFSET_STEP_SAMPLES)
    index = np.arange(n_offsets)
    reach = np.minimum(index + back_steps, n_offsets - 1)
    back = np.zeros((len(sub_times), n_offsets), dtype=np.int32)
    score = emission(0)
    for i in range(1, len(sub_times)):
        running_max = np.maximum.accumulate(score)
        # Index of the best predecessor at or below each offset.
        running_arg = np.maximum.accumulate(np.where(score >= running_max, index, 0))
        jump = running_max[reach] - jump_penalty
        use_jump = jump > score
        back[i] = np.where(use_jump, running_arg[reach], index)
        score = np.where(use_jump, jump, score) + emission(i)

    k = int(np.argmax(score))
    objective = float(score[k])
    path = [0] * len(sub_times)
    for i in range(len(sub_times) - 1, -1, -1):
        path[i] = k
        k = int(back[i][k])
    offsets = [float(offsets_samples[k]) / SAMPLE_RATE for k in path]
    changes = sum(a != b for a, b in pairwise(path))
    net = sum(float(timing_score(i)[k]) for i, k in enumerate(path))
    return CutAlignment(
        offsets=offsets,
        segments=_segments(offsets, sub_times, ref_times),
        objective=objective,
        score_per_cue=(net - jump_penalty * changes) / len(path),
    )


def assess_cut_alignment(
    sub_times: Times,
    ref_times: Times,
    offsets: Sequence[float],
    *,
    min_uncovered_cues: int = 3,
    max_gap_seconds: float = 20.0,
) -> CutAssessment:
    """How well the shifted cues sit on the reference, and what stays uncovered.

    Uncovered reference dialogue is expected when the video is a longer cut: the
    added scenes have no translation. It is reported, not counted as a fault.
    """
    if not sub_times:
        return CutAssessment(0.0, 0.0, [])
    ref_starts = np.sort(np.array([start for start, _ in ref_times], dtype=float))
    shifted = [
        (start + off, end + off)
        for (start, end), off in zip(sub_times, offsets, strict=True)
    ]
    distances = _start_distances(np.array([s for s, _ in shifted]), ref_starts)

    spans = sorted(shifted)
    span_starts = np.array([s for s, _ in spans])
    span_ends = np.maximum.accumulate(np.array([e for _, e in spans]))
    uncovered: list[tuple[float, float, int]] = []
    run: list[tuple[float, float]] = []

    def flush() -> None:
        if len(run) >= min_uncovered_cues:
            uncovered.append((run[0][0], run[-1][1], len(run)))
        run.clear()

    for start, end in sorted(ref_times):
        # Covered when some shifted cue overlaps it, with a second of slack.
        pos = int(np.searchsorted(span_starts, end + 1.0))
        covered = pos > 0 and span_ends[pos - 1] > start - 1.0
        if covered:
            continue
        if run and start - run[-1][1] > max_gap_seconds:
            flush()
        run.append((start, end))
    flush()

    return CutAssessment(
        within_half_second=float((distances <= 0.5).mean()),
        within_one_second=float((distances <= 1.0).mean()),
        uncovered=uncovered,
    )


def review_windows(
    alignment: CutAlignment,
    *,
    context: int = 4,
    min_jump_seconds: float = 1.0,
    min_segment_cues: int = 6,
    min_agreement: float = 0.6,
    head: int = 10,
) -> list[tuple[int, int]]:
    """Cue ranges (0-based, inclusive) where the timing score needs a second look.

    These are the places a cut-aware alignment goes wrong: around every jump,
    short segments, segments whose cues miss the reference cue starts, and the
    opening recap. Overlapping ranges are merged.
    """
    n = len(alignment.offsets)
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    segments = alignment.segments
    if len(segments) > 1 and head > 0:
        ranges.append((0, min(head, n) - 1))
    for prev, seg in pairwise(segments):
        if abs(seg.offset_seconds - prev.offset_seconds) >= min_jump_seconds:
            ranges.append(
                (max(seg.first - context, 0), min(seg.first + context, n) - 1)
            )
    for seg in segments:
        short = len(segments) > 1 and seg.last - seg.first + 1 < min_segment_cues
        if short or seg.agreement < min_agreement:
            ranges.append((max(seg.first - 1, 0), min(seg.last + 1, n - 1)))

    merged: list[tuple[int, int]] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged
