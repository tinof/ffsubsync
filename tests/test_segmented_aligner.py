"""Tests for SegmentedAligner voting and confidence gate."""

import numpy as np
import pytest

from ffsubsync.aligners import (
    FailedToFindAlignmentException,
    FFTAligner,
    MaxScoreAligner,
    SegmentedAligner,
)


def _make_speech_signal(
    length: int, speech_fraction: float = 0.3, seed: int = 42
) -> np.ndarray:
    """Synthetic binary speech signal."""
    rng = np.random.default_rng(seed)
    return rng.choice(
        [0, 1], size=length, p=[1 - speech_fraction, speech_fraction]
    ).astype(float)


def _shift_signal(sig: np.ndarray, offset: int) -> np.ndarray:
    """Shift signal by offset samples (positive = shift right)."""
    if offset == 0:
        return sig.copy()
    result = np.zeros_like(sig)
    if offset > 0:
        result[offset:] = sig[: len(sig) - offset]
    else:
        result[: len(sig) + offset] = sig[-offset:]
    return result


def _make_segmented_aligner(
    window_seconds: int = 20, overlap_seconds: int = 10
) -> SegmentedAligner:
    return SegmentedAligner(
        window_size_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate=100,
        tolerance_seconds=0.5,
    )


class TestSegmentedAlignerHighAgreement:
    """All windows agree → accept with high/medium confidence."""

    def test_three_windows_agree_accepts(self):
        """3/3 windows voting same bin → accepted."""
        ref = _make_speech_signal(3000)
        # subtitle is 2s ahead in all windows
        offset = 200
        sub = _shift_signal(ref, offset)

        aligner = _make_segmented_aligner()
        aligner.fit(ref, sub, get_score=True)

        assert aligner.confidence_ in ("high", "medium")
        assert aligner.vote_ratio_ >= 0.5
        assert aligner.best_offset_ is not None

    def test_confidence_high_when_all_agree(self):
        """Perfect agreement → confidence == 'high'."""
        ref = _make_speech_signal(2500, seed=7)
        sub = _shift_signal(ref, 100)
        aligner = _make_segmented_aligner()
        aligner.fit(ref, sub)
        assert aligner.confidence_ == "high"
        assert aligner.vote_ratio_ == 1.0


class TestSegmentedAlignerLowAgreement:
    """Windows disagree → raise FailedToFindAlignmentException."""

    def test_all_windows_disagree_raises(self):
        """Each window proposes a wildly different offset → rejected."""
        # Build ref: three segments back-to-back (3000 samples = 3x 10s windows at 100 Hz)
        ref = _make_speech_signal(3000, seed=1)

        # Build a sub that has different offsets in each window section
        sub = np.zeros(3000)
        sub[:1000] = _shift_signal(ref[:1000], 300)  # +3s in window 0
        sub[1000:2000] = _shift_signal(ref[1000:2000], -200)  # -2s in window 1
        sub[2000:] = _shift_signal(ref[2000:], 100)  # +1s in window 2

        aligner = SegmentedAligner(
            window_size_seconds=10,
            overlap_seconds=0,
            sample_rate=100,
            tolerance_seconds=0.5,
        )
        with pytest.raises(FailedToFindAlignmentException, match="low confidence"):
            aligner.fit(ref, sub)

    def test_low_confidence_attribute_set(self):
        """After failed majority vote, confidence_ == 'low'."""
        ref = _make_speech_signal(3000, seed=2)
        # Deliberately mismatched sub-signals per window
        sub = np.concatenate(
            [
                _shift_signal(ref[:1000], 400),
                _shift_signal(ref[1000:2000], -300),
                _shift_signal(ref[2000:], 200),
            ]
        )
        aligner = SegmentedAligner(
            window_size_seconds=10,
            overlap_seconds=0,
            sample_rate=100,
            tolerance_seconds=0.5,
        )
        import contextlib

        with contextlib.suppress(FailedToFindAlignmentException):
            aligner.fit(ref, sub)
        assert aligner.confidence_ == "low"


class TestSegmentedAlignerShortInput:
    """Input shorter than window → falls back to FFTAligner transparently."""

    def test_short_input_fallback(self):
        """Short input (< window) falls back to FFTAligner without raising."""
        ref = _make_speech_signal(500)  # 5s < 10s window
        sub = _shift_signal(ref, 50)

        aligner = SegmentedAligner(
            window_size_seconds=10,
            overlap_seconds=5,
            sample_rate=100,
        )
        aligner.fit(ref, sub)

        assert aligner.best_offset_ is not None
        assert aligner.confidence_ == "high"

    def test_short_input_offset_matches_fft(self):
        """Fallback result matches direct FFTAligner."""
        ref = _make_speech_signal(400, seed=9)
        offset = -30
        sub = _shift_signal(ref, offset)

        fft_result = FFTAligner().fit_transform(ref, sub)
        seg_result = SegmentedAligner(
            window_size_seconds=10, sample_rate=100
        ).fit_transform(ref, sub)

        assert seg_result == fft_result


class TestMaxScoreAlignerLowConfidenceHandling:
    """MaxScoreAligner swallows low-confidence SegmentedAligner results gracefully."""

    def test_all_subpipes_low_confidence_raises(self):
        """If all SegmentedAligner subpipes fail confidence gate, MaxScoreAligner raises."""
        # Force a signal where windows will disagree: use very short window
        ref = _make_speech_signal(3000, seed=3)
        sub_disagreeing = np.concatenate(
            [
                _shift_signal(ref[:1000], 400),
                _shift_signal(ref[1000:2000], -300),
                _shift_signal(ref[2000:], 200),
            ]
        )

        aligner = MaxScoreAligner(
            SegmentedAligner,
            srtin=None,
            window_size_seconds=10,
            overlap_seconds=0,
        )
        with pytest.raises(FailedToFindAlignmentException):
            aligner.fit_transform(ref, sub_disagreeing)


class TestSegmentedAlignerIntegration:
    """End-to-end: SegmentedAligner correctly recovers offset when windows agree."""

    def test_recovers_known_offset(self):
        """Recover a 5s offset from a signal long enough for 3 windows."""
        ref = _make_speech_signal(6000, speech_fraction=0.4, seed=20)
        # _shift_signal(ref, +N) moves content N samples right → aligner returns -N.
        shift = 500  # 5s at sample_rate=100
        sub = _shift_signal(ref, shift)

        aligner = SegmentedAligner(
            window_size_seconds=20,
            overlap_seconds=10,
            sample_rate=100,
            tolerance_seconds=1.0,
        )
        result = aligner.fit_transform(ref, sub)
        # |detected offset| should be within 1s of the true shift magnitude.
        assert abs(abs(result) - shift) <= 100
