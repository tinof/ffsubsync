"""Tests for the split-penalty (alass-style) aligner.

The first group is ported from upstream tests/test_split_aligner.py. The rest
cover the fork's changes: residual search around a global offset, orphan-cue
repair at negative jumps, and the engine and ssync wiring.
"""

import types
from datetime import timedelta

import numpy as np
import pytest
import srt

from ffsubsync import ffsubsync
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile
from ffsubsync.split_aligner import (
    compute_split_offsets,
    enforce_cue_order,
    split_segments,
)
from ffsubsync.subtitle_parser import GenericSubtitleParser
from ffsubsync.subtitle_transformers import VariableSubtitleShifter


def _cue(start_s, end_s, content="hello world"):
    inner = srt.Subtitle(
        index=1,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )
    return GenericSubtitle(inner.start, inner.end, inner)


def _reference(intervals, total_seconds):
    arr = np.zeros(int(total_seconds * SAMPLE_RATE) + 2, dtype=float)
    for a, b in intervals:
        arr[round(a * SAMPLE_RATE) : round(b * SAMPLE_RATE)] = 1.0
    return arr


def _num_splits(offsets):
    return sum(1 for i in range(1, len(offsets)) if offsets[i] != offsets[i - 1])


def _align(reference, cues, split_penalty, max_offset_seconds=7, **kwargs):
    offsets, _ = compute_split_offsets(
        reference,
        cues,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=split_penalty,
        max_offset_samples=int(max_offset_seconds * SAMPLE_RATE),
        **kwargs,
    )
    return offsets


# ---- ported from upstream -------------------------------------------------


def test_recovers_mid_file_break():
    # First half at offset 0, second half needs +6 s; no single offset fits both.
    reference = _reference(
        [(1.0, 1.4), (3.0, 3.9), (17.0, 17.6), (19.0, 20.0)], total_seconds=22
    )
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9), _cue(11.0, 11.6), _cue(13.0, 14.0)]
    offsets = _align(reference, cues, split_penalty=0.5 * SAMPLE_RATE)
    assert offsets == [0, 0, 6 * SAMPLE_RATE, 6 * SAMPLE_RATE]
    assert _num_splits(offsets) == 1


def test_collapses_to_single_offset():
    reference = _reference(
        [(6.0, 6.4), (8.0, 8.9), (10.0, 10.6), (12.0, 13.0)], total_seconds=14
    )
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9), _cue(5.0, 5.6), _cue(7.0, 8.0)]
    offsets = _align(reference, cues, split_penalty=1e9)
    assert offsets == [5 * SAMPLE_RATE] * 4


def test_already_aligned_is_noop():
    reference = _reference([(1.0, 1.4), (3.0, 3.9)], total_seconds=6)
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9)]
    assert _align(reference, cues, split_penalty=0.5 * SAMPLE_RATE) == [0, 0]


def test_length_penalty_prefers_matching_block():
    # Pure overlap is flat inside the long block at +3 s and tie-breaks there; the
    # guard term prefers the exactly-sized, isolated block at +10 s.
    reference = _reference([(3.0, 8.0), (10.0, 11.0)], total_seconds=13)
    cues = [_cue(0.0, 1.0)]
    assert _align(reference, cues, 1e9, max_offset_seconds=12) == [3 * SAMPLE_RATE]
    penalized = _align(reference, cues, 1e9, max_offset_seconds=12, length_penalty=0.25)
    assert penalized == [10 * SAMPLE_RATE]


def test_empty_input():
    offsets, score = compute_split_offsets(
        _reference([(1, 2)], total_seconds=3),
        [],
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=1.0,
        max_offset_samples=100,
    )
    assert offsets == []
    assert score == 0.0


def test_non_dialogue_cue_follows_its_neighbours():
    reference = _reference([(3.0, 3.9), (17.0, 17.6)], total_seconds=20)
    cues = [_cue(3.0, 3.9), _cue(5.0, 6.0, "♪ ♪"), _cue(11.0, 11.6)]
    offsets = _align(reference, cues, split_penalty=0.3 * SAMPLE_RATE)
    # The music cue has no rating, so it costs nothing to keep a neighbour's offset.
    assert offsets[1] in (offsets[0], offsets[2])


def test_probability_reference_is_thresholded():
    # Regression: a reference may hold probabilities. With "> 0" every silence frame at
    # 0.01 counted as speech and correctly placed cues were moved.
    reference = np.full(40 * SAMPLE_RATE, 0.01)
    for a, b in [(10, 11), (20, 22), (30, 31.5)]:
        reference[round(a * SAMPLE_RATE) : round(b * SAMPLE_RATE)] = 0.99
    cues = [_cue(10, 11), _cue(20, 22), _cue(30, 31.5)]
    offsets = _align(
        reference, cues, split_penalty=30 * SAMPLE_RATE, max_offset_seconds=5
    )
    assert offsets == [0, 0, 0]


# ---- fork: residual search -------------------------------------------------


def test_start_seconds_turns_offsets_into_residuals():
    # Cues are 40 s early; with the global offset folded into start_seconds the
    # DP only has to find the remaining +0/+3 s, well inside a 5 s window.
    reference = _reference([(41.0, 41.5), (43.0, 44.0), (53.0, 53.7)], 60)
    cues = [_cue(1.0, 1.5), _cue(3.0, 4.0), _cue(10.0, 10.7)]
    residuals, _ = compute_split_offsets(
        reference,
        cues,
        sample_rate=SAMPLE_RATE,
        start_seconds=-40.0,
        split_penalty=0.3 * SAMPLE_RATE,
        max_offset_samples=5 * SAMPLE_RATE,
    )
    assert residuals == [0, 0, 3 * SAMPLE_RATE]


def _synthetic_episode(jumps, total_seconds=1200, seed=7):
    """Cues with random timing; ``jumps`` maps a video time to an inserted gap.

    Returns (cues in subtitle time, reference speech in video time, true offsets).
    """
    rng = np.random.default_rng(seed)
    cues, truth, intervals = [], [], []
    t = 5.0
    while t < total_seconds - 60:
        duration = float(rng.uniform(0.8, 3.0))
        shift = sum(gap for at, gap in jumps if t >= at)
        cues.append(_cue(t, t + duration))
        truth.append(shift * SAMPLE_RATE)
        intervals.append((t + shift, t + shift + duration))
        t += duration + float(rng.uniform(0.4, 4.0))
    return cues, _reference(intervals, total_seconds + 120), truth


def test_recovers_two_close_ad_breaks():
    # Two 20 s breaks 60 s apart. Windows of 200 s cannot separate them.
    cues, reference, truth = _synthetic_episode([(600, 20), (660, 20)])
    offsets = _align(
        reference, cues, split_penalty=5 * SAMPLE_RATE, max_offset_seconds=45
    )
    assert offsets == truth
    assert _num_splits(offsets) == 2


# ---- fork: negative jumps ---------------------------------------------------


def test_enforce_cue_order_moves_orphans_after_previous_segment():
    cues = [_cue(0, 1), _cue(2, 3), _cue(4, 5), _cue(6, 7)]
    offsets = [0.0, 0.0, -3.0 * SAMPLE_RATE, -3.0 * SAMPLE_RATE]
    fixed = enforce_cue_order(cues, offsets, sample_rate=SAMPLE_RATE)
    # Cue 2 would start at 1 s, inside cue 1 (2-3 s): it moves to start at 3 s.
    assert fixed[2] == -1.0 * SAMPLE_RATE
    # Cue 3 already starts at 3 s, so the correct cues keep their offset.
    assert fixed[3] == -3.0 * SAMPLE_RATE
    starts = [
        c.start.total_seconds() * SAMPLE_RATE + o
        for c, o in zip(cues, fixed, strict=True)
    ]
    assert starts == sorted(starts)


def test_enforce_cue_order_leaves_positive_jumps_alone():
    cues = [_cue(0, 1), _cue(2, 3), _cue(4, 5)]
    offsets = [0.0, 5.0 * SAMPLE_RATE, 5.0 * SAMPLE_RATE]
    assert enforce_cue_order(cues, offsets, sample_rate=SAMPLE_RATE) == offsets


def test_split_segments_summary():
    assert split_segments([0, 0, 600, 600, 600], SAMPLE_RATE) == [(2, 0.0), (3, 6.0)]


# ---- shifter ----------------------------------------------------------------


def _subs_file(cues):
    return GenericSubtitlesFile(cues, sub_format="srt", encoding="utf-8")


def test_variable_shifter_applies_one_offset_per_cue():
    shifted = VariableSubtitleShifter([1.0, -0.5]).fit_transform(
        _subs_file([_cue(2, 3), _cue(10, 11)])
    )
    assert [s.start.total_seconds() for s in shifted] == [3.0, 9.5]
    assert [s.end.total_seconds() for s in shifted] == [4.0, 10.5]


def test_variable_shifter_clamps_at_zero_and_checks_length():
    shifted = VariableSubtitleShifter([-5.0]).fit_transform(_subs_file([_cue(2, 3)]))
    assert shifted[0].start.total_seconds() == 0.0
    with pytest.raises(ValueError, match="one offset per cue"):
        VariableSubtitleShifter([0.0]).fit(_subs_file([_cue(0, 1), _cue(2, 3)]))


# ---- engine wiring -----------------------------------------------------------


def test_cli_parses_split_flags():
    parse = ffsubsync.make_parser().parse_args
    assert parse(["movie.mkv", "--split-penalty"]).split_penalty == 30.0
    assert parse(["movie.mkv", "--split-penalty", "2.5"]).split_penalty == 2.5
    assert parse(["movie.mkv"]).split_penalty is None


def test_split_and_piecewise_audio_are_exclusive():
    args = ffsubsync.make_parser().parse_args(
        ["movie.mkv", "-i", "in.srt", "--split-penalty", "--piecewise-audio"]
    )
    with pytest.raises(ValueError, match="exclusive"):
        ffsubsync.validate_args(args)


def test_try_sync_split_mode_corrects_a_jump(tmp_path):
    # Global FFT finds one offset (-4 s for the whole file); split mode then fixes
    # the second part, which is +6 s later after an inserted break.
    cues, reference, truth = _synthetic_episode([(300, 6)], total_seconds=600)
    global_shift = -4.0
    srtin = tmp_path / "in.srt"
    srtin.write_text(
        srt.compose(
            [
                srt.Subtitle(
                    i + 1,
                    c.start - timedelta(seconds=global_shift),
                    c.end - timedelta(seconds=global_shift),
                    f"line {i}",
                )
                for i, c in enumerate(cues)
            ]
        )
    )
    srtout = tmp_path / "out.srt"
    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(srtin), "-o", str(srtout), "--split-penalty"]
    )
    args.skip_infer_framerate_ratio = True
    args.auto_sync = False
    reference_pipe = types.SimpleNamespace(transform=lambda _: reference)
    result = {"retval": 0}

    assert ffsubsync.try_sync(args, reference_pipe, result) is True

    parser = GenericSubtitleParser()
    parser.fit(str(srtout))
    got = [s.start.total_seconds() for s in parser.subs_]
    want = [
        c.start.total_seconds() + t / SAMPLE_RATE
        for c, t in zip(cues, truth, strict=True)
    ]
    assert np.allclose(got, want, atol=0.02)
    assert len(result["split_segments"]) == 2
