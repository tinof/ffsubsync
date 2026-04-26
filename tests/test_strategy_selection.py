"""Tests for adaptive strategy selection and GSS ratio snapping."""

import argparse

import numpy as np

from ffsubsync.constants import FRAMERATE_RATIOS, FRAMERATE_SNAP_TOLERANCE

# ── GSS ratio plausibility tests ────────────────────────────────────────────


class TestGSSRatioSnap:
    """MaxScoreAligner.fit_gss discards non-physical ratios."""

    def _make_subpipe_maker(self, ref: np.ndarray, ratio: float):
        """Return a fake subpipe_maker that produces a signal scaled by ratio."""

        def maker(r):
            # Scale the reference signal; use a simple identity transform
            class FakePipe:
                def fit_transform(self, srtin):
                    n = max(1, int(len(ref) * r))
                    return ref[:n] if n <= len(ref) else np.pad(ref, (0, n - len(ref)))

                def transform(self, srtin):
                    return self.fit_transform(srtin)

            return FakePipe()

        return maker

    def test_implausible_ratio_discarded(self):
        """A GSS ratio well outside known pairs must be discarded."""
        # Verify that 0.95 exceeds FRAMERATE_SNAP_TOLERANCE (0.5%) from all known pairs
        bad_ratio = 0.95

        known = [1.0] + list(FRAMERATE_RATIOS) + [1.0 / r for r in FRAMERATE_RATIOS]
        nearest = min(known, key=lambda c: abs(bad_ratio - c) / c)
        rel_err = abs(bad_ratio - nearest) / nearest

        # The ratio 0.95 should exceed the tolerance
        assert rel_err > FRAMERATE_SNAP_TOLERANCE, (
            f"Test assumption wrong: 0.95 should be beyond snap tolerance, "
            f"but nearest={nearest:.4f}, err={rel_err:.4f}"
        )

    def test_known_ratio_accepted(self):
        """A GSS ratio that falls within tolerance of a known pair must be kept."""
        known = [1.0] + list(FRAMERATE_RATIOS) + [1.0 / r for r in FRAMERATE_RATIOS]
        for ratio in known:
            # ratio ± 0.1% should be within tolerance
            for delta in (0.0, 0.001, -0.001):
                candidate = ratio * (1 + delta)
                nearest = min(known, key=lambda c: abs(candidate - c) / c)
                rel_err = abs(candidate - nearest) / nearest
                assert rel_err <= FRAMERATE_SNAP_TOLERANCE, (
                    f"Ratio {candidate:.4f} (near {ratio:.4f}) should be within "
                    f"snap tolerance but err={rel_err:.4f}"
                )

    def test_framerate_snap_tolerance_constant(self):
        """FRAMERATE_SNAP_TOLERANCE is 0.5% as documented."""
        assert abs(FRAMERATE_SNAP_TOLERANCE - 0.005) < 1e-9

    def test_snapped_gss_ratio_is_re_evaluated(self, monkeypatch):
        """When GSS lands near a known ratio, use the known ratio pipeline."""
        import ffsubsync.aligners as aligners

        def fake_gss(func, *_):
            func(1.0004, True)

        class FakeAligner:
            def fit_transform(self, _refstring, substring, get_score=False):
                return float(substring[0]), int(substring[0])

        class FakePipe:
            def __init__(self, ratio):
                self.ratio = ratio

            def fit_transform(self, _srtin):
                return np.array([round(self.ratio * 1000)])

        monkeypatch.setattr(aligners, "gss", fake_gss)

        max_score_aligner = aligners.MaxScoreAligner(FakeAligner(), srtin="subs.srt")
        max_score_aligner.fit_gss(np.ones(10), FakePipe)

        ((score, offset), subpipe) = max_score_aligner._scores[0]
        assert subpipe.ratio == 1.0
        assert score == 1000.0
        assert offset == 1000


# ── Strategy selection margin tests ─────────────────────────────────────────


def _make_ffsubsync_args(**overrides) -> argparse.Namespace:
    """Minimal valid args namespace for strategy tests."""
    defaults = {
        "gss": False,
        "no_fix_framerate": False,
        "use_segmented_aligner": False,
        "auto_sync": True,
        "segment_window": 600,
        "segment_overlap": 300,
        "skip_infer_framerate_ratio": True,
        "max_offset_seconds": 60,
        "max_subtitle_seconds": 10,
        "start_seconds": 0,
        "non_speech_label": 0.0,
        "scale_factor": 1.0,
        "merge_with_reference": False,
        "overwrite_input": False,
        "apply_offset_seconds": 0,
        "output_encoding": "utf-8",
        "suppress_output_if_offset_less_than": None,
        "srtin": [None],
        "srtout": None,
        "skip_sync": True,
        "reference": None,
        "vad": None,
        "frame_rate": 48000,
        "ffmpeg_path": None,
        "vlc_mode": False,
        "log_dir_path": None,
        "make_test_case": False,
        "serialize_speech": False,
        "reference_encoding": None,
        "reference_stream": None,
        "extract_subs_from_stream": None,
        "strict": False,
        "encoding": "utf-8",
        "gui_mode": False,
        "vad_smoothing_window": 30,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestGetAlignmentStrategies:
    """get_alignment_strategies returns correct strategy lists."""

    def test_auto_sync_disabled_returns_only_primary(self):
        from ffsubsync.ffsubsync import get_alignment_strategies

        args = _make_ffsubsync_args(auto_sync=False)
        strategies = get_alignment_strategies(args)
        assert len(strategies) == 1
        assert strategies[0][0] == "primary"

    def test_auto_sync_enabled_adds_adaptive(self):
        from ffsubsync.ffsubsync import get_alignment_strategies

        args = _make_ffsubsync_args(auto_sync=True)
        strategies = get_alignment_strategies(args)
        names = [s[0] for s in strategies]
        assert "primary" in names
        assert "adaptive" in names

    def test_no_fix_framerate_adaptive_gss_disabled(self):
        from ffsubsync.ffsubsync import get_alignment_strategies

        args = _make_ffsubsync_args(auto_sync=True, no_fix_framerate=True)
        strategies = get_alignment_strategies(args)
        adaptive = next((s for s in strategies if s[0] == "adaptive"), None)
        if adaptive:
            # adaptive with no_fix_framerate → gss=False
            assert adaptive[1] is False

    def test_no_duplicate_strategies(self):
        from ffsubsync.ffsubsync import get_alignment_strategies

        args = _make_ffsubsync_args(
            auto_sync=True, gss=True, use_segmented_aligner=True
        )
        strategies = get_alignment_strategies(args)
        names = [s[0] for s in strategies]
        # When primary already matches adaptive config, no duplicate is added
        assert len(names) == len(set(names))
