"""alass-style split-penalty alignment.

Ported from upstream ffsubsync (c33e10b, 7c5f9c1), which implements the core idea
of `alass <https://github.com/kaegi/alass>`_: every cue may take its own offset,
but each change of offset between consecutive cues costs a *split penalty*. The
dynamic program maximizes::

    sum_i rating(cue_i @ offset_i)  -  split_penalty * (number of splits)

and returns a piecewise-constant offset per cue. With a large penalty the optimum
collapses to one global offset.

This suits discrete edit jumps (an ad break, a cut or added scene, two discs
joined into one file): the jump lands exactly between two cues, and a short
inserted segment can be found. Slow progressive drift is better handled by the
anchor interpolation in :mod:`ffsubsync.piecewise`, because a staircase of
constant offsets can only approximate a slope.

Rating
------
The rating of cue ``[start, end]`` at offset ``o`` is its overlap with the
reference speech, an O(1) prefix-sum lookup::

    rating(cue, o) = ref_cumsum[end + o] - ref_cumsum[start + o]

minus ``length_penalty`` times the reference speech in a guard band just outside
the cue's edges. The guard term rewards cue edges that line up with speech
boundaries, so a short cue can be placed inside a long speech block and a
same-length block beats a longer one.

The prefix sum is interpolated linearly, which is exact for a 0/1 reference, so
the offset grid can also be finer than one sample (``offset_step_samples``). The
engine keeps it at one sample; 10 ms is below what VAD boundaries resolve.

Differences from upstream
-------------------------
* Callers center the offset window on the global FFT offset of each framerate
  scale (via ``start_seconds``), instead of on zero, so a jump is measured from
  the global fit. The ``n_cues x n_offsets`` back-pointer table is int32, half
  of upstream's int64 (about 48 MB for 1000 cues at +-60 s).
* :func:`enforce_cue_order` repairs negative jumps, where cues for content that
  is missing from the video would otherwise start inside the previous segment.
"""

import logging

import numpy as np

from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.speech_transformers import _is_metadata

logger: logging.Logger = logging.getLogger(__name__)


# Upper bound (in samples) on the length-penalty guard band. The guard is normally
# the cue's own duration, but a very long cue would otherwise reach across
# neighbouring dialogue and penalize a good placement (2 s at 100 Hz).
_MAX_GUARD_SAMPLES: int = 200


def _cue_sample_bounds(
    cues: list[GenericSubtitle], sample_rate: int, start_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cue ``(start_sample, end_sample, is_speech)`` arrays.

    Bounds stay floats so a sub-sample offset grid can use their exact timing.
    The ``_is_metadata`` gate matches :class:`SubtitleSpeechTransformer`, so
    non-dialogue cues contribute no rating and the DP leaves them on a
    neighbour's offset for free.
    """
    n = len(cues)
    starts = np.zeros(n, dtype=np.float64)
    ends = np.zeros(n, dtype=np.float64)
    is_speech = np.zeros(n, dtype=bool)
    for i, sub in enumerate(cues):
        start = (sub.start.total_seconds() - start_seconds) * sample_rate
        duration = (sub.end.total_seconds() - sub.start.total_seconds()) * sample_rate
        starts[i] = start
        ends[i] = start + duration
        is_speech[i] = not _is_metadata(sub.content, i == 0 or i + 1 == n)
    return starts, ends, is_speech


def compute_split_offsets(
    reference: np.ndarray,
    cues: list[GenericSubtitle],
    *,
    sample_rate: int,
    start_seconds: float,
    split_penalty: float,
    max_offset_samples: int,
    length_penalty: float = 0.0,
    offset_step_samples: float = 1.0,
) -> tuple[list[float], float]:
    """Return ``(offsets, score)``: one offset in samples per cue, and the objective.

    ``reference`` is the binary reference speech signal at ``sample_rate`` Hz and
    ``cues`` the (already framerate-scaled) subtitle cues. ``start_seconds`` is
    subtracted from cue times; callers pass ``start_seconds - global_offset`` so
    the returned offsets are residuals around that global offset.
    ``split_penalty`` is in overlap samples. Offsets range over
    ``[-max_offset_samples, +max_offset_samples]`` in steps of
    ``offset_step_samples``. ``score`` compares alignments of the same cues,
    e.g. across framerate scales; it is not comparable to FFT scores.
    """
    n = len(cues)
    if n == 0:
        return [], 0.0

    starts, ends, is_speech = _cue_sample_bounds(cues, sample_rate, start_seconds)
    guards = np.minimum(ends - starts, float(_MAX_GUARD_SAMPLES))

    step = float(offset_step_samples)
    if step <= 0:
        step = 1.0
    n_off = round(2 * max_offset_samples / step) + 1
    offsets = -float(max_offset_samples) + step * np.arange(n_off)

    # Prefix sum of the binary reference. Pad both ends by the offset half-width
    # plus the max guard band, so every shifted lookup lands in a flat region
    # (0 on the left, total speech on the right) that adds no speech.
    # >= 0.5, not > 0: a serialized reference may hold probabilities, and a
    # silence probability of 0.01 must not count as speech.
    ref = (np.asarray(reference, dtype=np.float64) >= 0.5).astype(np.float64)
    cumsum = np.concatenate([np.zeros(1), np.cumsum(ref)])  # cumsum[k] = sum(ref[:k])
    # Cue positions can lie outside the reference (start_seconds carries the
    # global offset), so pad by the cue extent as well as the offset window.
    cue_extent = max(0.0, -float(starts.min()), float(ends.max()) - len(ref))
    pad = int(max_offset_samples + cue_extent) + _MAX_GUARD_SAMPLES + 2
    padded = np.concatenate([np.zeros(pad), cumsum, np.full(pad, cumsum[-1])])
    max_pos = len(padded) - 1

    def _interp(pos: np.ndarray) -> np.ndarray:
        # cumsum[k] + ref[k] * frac is the exact integral up to a fractional
        # position of a 0/1 step function, so this is exact, not approximate.
        pos = np.clip(pos, 0.0, float(max_pos))
        lo = np.floor(pos).astype(np.int64)
        np.clip(lo, 0, max_pos - 1, out=lo)
        frac = pos - lo
        return padded[lo] * (1.0 - frac) + padded[lo + 1] * frac

    def _rating_row(i: int) -> np.ndarray:
        if not is_speech[i]:
            return np.zeros(n_off)
        start_pos = starts[i] + offsets + pad
        end_pos = ends[i] + offsets + pad
        cs = _interp(start_pos)
        ce = _interp(end_pos)
        overlap = ce - cs
        if length_penalty and guards[i] > 0:
            gl = cs - _interp(start_pos - guards[i])  # speech just before the cue
            gr = _interp(end_pos + guards[i]) - ce  # speech just after the cue
            return overlap - length_penalty * (gl + gr)
        return overlap

    # Forward DP. dp[o] = best total rating for cues[0..i] with cue i at offset o.
    # back[i, o] is -1 when cue i kept the previous cue's offset, otherwise the
    # offset index the previous cue jumped from (the argmax of the previous row).
    dp = _rating_row(0)
    back = np.full((n, n_off), -1, dtype=np.int32)  # row 0 unused
    for i in range(1, n):
        best_prev_idx = int(np.argmax(dp))
        jump_value = dp[best_prev_idx] - split_penalty
        jumped = jump_value > dp
        dp = _rating_row(i) + np.where(jumped, jump_value, dp)
        back[i] = np.where(jumped, best_prev_idx, -1)

    best_total = float(dp.max())

    offset_idx = np.empty(n, dtype=np.int64)
    cur = int(np.argmax(dp))
    for i in range(n - 1, -1, -1):
        offset_idx[i] = cur
        if i > 0:
            b = int(back[i, cur])
            cur = b if b >= 0 else cur

    return [float(offsets[idx]) for idx in offset_idx], best_total


def enforce_cue_order(
    cues: list[GenericSubtitle],
    offsets: list[float],
    *,
    sample_rate: int,
) -> list[float]:
    """Keep cues after a negative jump from starting inside the previous segment.

    At a negative jump (content that the subtitle has is missing from the video)
    the cues for the missing content would start before the previous segment's
    last cue ends. Those orphan cues are moved to start at that end. The first
    cue that already starts after it, and everything following, keeps its offset,
    so correct cues are never pushed later. Orphans may overlap each other; their
    video does not exist, so no position for them is right.
    """
    if len(cues) < 2:
        return list(offsets)
    starts, ends, _ = _cue_sample_bounds(cues, sample_rate, 0.0)
    fixed = list(offsets)
    for i in range(1, len(cues)):
        if offsets[i] >= offsets[i - 1]:
            continue
        anchor = ends[i - 1] + fixed[i - 1]
        j = i
        while (
            j < len(cues) and offsets[j] == offsets[i] and starts[j] + fixed[j] < anchor
        ):
            fixed[j] = anchor - starts[j]
            j += 1
    return fixed


def split_segments(offsets: list[float], sample_rate: int) -> list[tuple[int, float]]:
    """Summarize per-cue offsets as ``(cue_count, offset_seconds)`` runs."""
    segments: list[tuple[int, float]] = []
    for off in offsets:
        seconds = off / float(sample_rate)
        if segments and segments[-1][1] == seconds:
            segments[-1] = (segments[-1][0] + 1, seconds)
        else:
            segments.append((1, seconds))
    return segments


def log_split_segments(offsets: list[float], sample_rate: int) -> None:
    """Log a human-readable summary of the piecewise offset function."""
    segments = split_segments(offsets, sample_rate)
    if not segments:
        return
    logger.info(
        "split alignment: %d segment(s), %d split(s)", len(segments), len(segments) - 1
    )
    for count, seconds in segments:
        logger.info("  %d cue(s) offset %.3fs", count, seconds)
