"""Tests for audio-based piecewise drift correction."""

import contextlib
from datetime import timedelta

import numpy as np

from ffsubsync.aligners import SegmentedAligner
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile
from ffsubsync.piecewise import (
    Anchor,
    WindowOffset,
    build_anchors,
    compute_window_offsets,
)
from ffsubsync.subtitle_transformers import PiecewiseSubtitleShifter


def _make_speech_signal(
    length: int, speech_fraction: float = 0.3, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(
        [0, 1], size=length, p=[1 - speech_fraction, speech_fraction]
    ).astype(float)


def _shift_signal(sig: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return sig.copy()
    result = np.zeros_like(sig)
    if offset > 0:
        result[offset:] = sig[: len(sig) - offset]
    else:
        result[: len(sig) + offset] = sig[-offset:]
    return result


def _drift_signal(sig: np.ndarray, offsets_by_window: list[int]) -> np.ndarray:
    """Shift successive equal chunks by increasing offsets."""
    chunk = len(sig) // len(offsets_by_window)
    out = np.zeros_like(sig)
    for i, offset in enumerate(offsets_by_window):
        lo, hi = (
            i * chunk,
            (i + 1) * chunk if i + 1 < len(offsets_by_window) else len(sig),
        )
        piece = sig[lo:hi]
        dest_lo = min(len(sig), max(0, lo + offset))
        dest_hi = min(len(sig), dest_lo + len(piece))
        out[dest_lo:dest_hi] = piece[: dest_hi - dest_lo]
    return out


class TestComputeWindowOffsets:
    def test_recovers_constant_offset_in_every_window(self):
        ref = _make_speech_signal(60000)
        sub = _shift_signal(ref, 150)

        windows = compute_window_offsets(
            ref, sub, window_size_seconds=100, overlap_seconds=50
        )

        assert len(windows) >= 3
        # The aligner reports the correction to apply: subs are 1.5s late here.
        for window in windows:
            assert abs(window.offset_seconds + 1.5) < 0.2

    def test_tracks_progressive_drift(self):
        ref = _make_speech_signal(120000)
        sub = _drift_signal(ref, [0, 100, 200, 300])

        windows = compute_window_offsets(
            ref, sub, window_size_seconds=200, overlap_seconds=100
        )
        anchors = build_anchors(windows)

        assert len(anchors) >= 3
        # Offsets should increase across the file, tracking the drift.
        assert anchors[-1].offset < anchors[0].offset

    def test_short_input_yields_no_windows(self):
        ref = _make_speech_signal(1000)
        sub = _shift_signal(ref, 10)

        assert compute_window_offsets(ref, sub, window_size_seconds=300) == []

    def test_silent_windows_are_skipped(self):
        ref = _make_speech_signal(60000)
        sub = _shift_signal(ref, 100)
        # Blank out the first half of the subtitle timeline.
        sub[:30000] = 0.0

        windows = compute_window_offsets(
            ref, sub, window_size_seconds=100, overlap_seconds=0, min_speech_seconds=20
        )

        assert windows
        assert all(w.center_seconds > 200 for w in windows)


class TestBuildAnchors:
    def test_returns_empty_for_too_few_windows(self):
        assert build_anchors([WindowOffset(10.0, 0.5, 100.0)]) == []

    def test_drops_low_scoring_windows(self):
        windows = [
            WindowOffset(100.0, 0.5, 1000.0),
            WindowOffset(200.0, 9.0, 1.0),
            WindowOffset(300.0, 0.6, 1000.0),
            WindowOffset(400.0, 0.7, 1000.0),
        ]

        anchors = build_anchors(windows)

        assert all(abs(a.offset) < 1.0 for a in anchors)

    def test_drops_offset_outliers(self):
        windows = [
            WindowOffset(100.0, 0.5, 1000.0),
            WindowOffset(200.0, 0.5, 1000.0),
            WindowOffset(300.0, 12.0, 1000.0),
            WindowOffset(400.0, 0.5, 1000.0),
            WindowOffset(500.0, 0.5, 1000.0),
        ]

        anchors = build_anchors(windows)

        assert all(a.offset < 1.0 for a in anchors)

    def test_enforces_monotonic_warp(self):
        # A backwards jump larger than the window spacing would reorder cues.
        windows = [
            WindowOffset(100.0, 0.0, 1000.0),
            WindowOffset(150.0, -80.0, 500.0),
            WindowOffset(200.0, 0.0, 1000.0),
        ]

        anchors = build_anchors(windows)

        times = [a.time for a in anchors]
        warped = [a.time + a.offset for a in anchors]
        assert times == sorted(times)
        assert warped == sorted(warped)


class TestPiecewiseSubtitleShifter:
    def _subs(self, starts):
        subs = [
            GenericSubtitle(
                timedelta(seconds=start), timedelta(seconds=start + 2.0), None
            )
            for start in starts
        ]
        return GenericSubtitlesFile(subs, sub_format="srt", encoding="utf-8")

    def test_interpolates_between_anchors(self):
        shifter = PiecewiseSubtitleShifter([(0.0, 0.0), (100.0, 10.0)])

        assert shifter.offset_at(50.0) == 5.0

    def test_extrapolates_flat_beyond_anchors(self):
        shifter = PiecewiseSubtitleShifter([(100.0, 2.0), (200.0, 4.0)])

        assert shifter.offset_at(0.0) == 2.0
        assert shifter.offset_at(1000.0) == 4.0

    def test_preserves_durations(self):
        subs = self._subs([10.0, 110.0])
        shifter = PiecewiseSubtitleShifter([(0.0, 1.0), (200.0, 5.0)])

        out = list(shifter.fit_transform(subs))

        for sub in out:
            assert abs((sub.end - sub.start).total_seconds() - 2.0) < 1e-9

    def test_applies_growing_offset(self):
        subs = self._subs([0.0, 100.0])
        shifter = PiecewiseSubtitleShifter([(0.0, 0.0), (100.0, 10.0)])

        out = list(shifter.fit_transform(subs))

        assert abs(out[0].start.total_seconds() - 0.0) < 1e-9
        assert abs(out[1].start.total_seconds() - 110.0) < 1e-9

    def test_clamps_negative_start_to_zero(self):
        subs = self._subs([1.0])
        shifter = PiecewiseSubtitleShifter([(0.0, -30.0), (100.0, -30.0)])

        out = list(shifter.fit_transform(subs))

        assert out[0].start.total_seconds() == 0.0


class TestSegmentedAlignerWindowResults:
    def test_records_windows_on_success(self):
        ref = _make_speech_signal(3000)
        sub = _shift_signal(ref, 200)
        aligner = SegmentedAligner(
            window_size_seconds=10, overlap_seconds=5, sample_rate=100
        )

        aligner.fit(ref, sub, get_score=True)

        assert len(aligner.window_results_) >= 3
        centers = [center for center, _, _ in aligner.window_results_]
        assert centers == sorted(centers)
        # Voting behavior is unchanged.
        assert abs(aligner.best_offset_ + 200) <= 50

    def test_records_windows_even_when_vote_fails(self):
        ref = _make_speech_signal(6000)
        # Each half shifted differently → no majority.
        sub = np.concatenate(
            [_shift_signal(ref[:3000], 100), _shift_signal(ref[3000:], -400)]
        )
        aligner = SegmentedAligner(
            window_size_seconds=10, overlap_seconds=5, sample_rate=100
        )

        with contextlib.suppress(Exception):
            aligner.fit(ref, sub, get_score=True)

        assert len(aligner.window_results_) > 0

    def test_anchors_from_anchor_dataclass_roundtrip(self):
        anchors = [Anchor(time=10.0, offset=1.0), Anchor(time=20.0, offset=2.0)]
        shifter = PiecewiseSubtitleShifter([(a.time, a.offset) for a in anchors])

        assert shifter.offset_at(15.0) == 1.5
