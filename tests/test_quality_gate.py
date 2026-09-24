"""Tests for the --skip-sync-on-low-quality safety net (port of upstream e2e3e4a)."""

import types

import numpy as np
import pytest

from ffsubsync import ffsubsync
from ffsubsync.constants import (
    DEFAULT_MAX_FRAMERATE_DEVIATION,
    DEFAULT_MIN_SCORE,
    DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    SAMPLE_RATE,
)
from ffsubsync.ffsubsync import assess_alignment_quality
from ffsubsync.subtitle_parser import GenericSubtitleParser

SRT = """1
00:00:10,000 --> 00:00:11,000
Hello

2
00:00:20,000 --> 00:00:21,000
World
"""

_THRESHOLDS = {
    "min_score": DEFAULT_MIN_SCORE,
    "max_offset_seconds": DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    "max_framerate_deviation": DEFAULT_MAX_FRAMERATE_DEVIATION,
}


# ---- pure assessment logic -----------------------------------------------


def test_quality_ok_for_plausible_alignment():
    assert assess_alignment_quality(500.0, 3.0, 1.0, **_THRESHOLDS) == []


def test_quality_rejects_negative_score():
    reasons = assess_alignment_quality(-1.0, 3.0, 1.0, **_THRESHOLDS)
    assert len(reasons) == 1
    assert "score" in reasons[0]


def test_default_offset_limit_allows_offsets_up_to_a_minute():
    # Subtitles from another release often need 30-60 s; the fork's default
    # must not reject them.
    assert assess_alignment_quality(500.0, 45.0, 1.0, **_THRESHOLDS) == []
    reasons = assess_alignment_quality(500.0, 75.0, 1.0, **_THRESHOLDS)
    assert len(reasons) == 1
    assert "offset" in reasons[0]


def test_default_framerate_threshold_allows_all_real_corrections():
    # The largest known correction is 25/23.976 ~= 1.0427.
    for scale in (1.0, 24.0 / 23.976, 25.0 / 24.0, 25.0 / 23.976, 24.0 / 25.0):
        assert assess_alignment_quality(500.0, 3.0, scale, **_THRESHOLDS) == []


def test_scale_within_snap_tolerance_of_a_known_ratio_is_accepted():
    assert assess_alignment_quality(500.0, 3.0, 1.0417 * 1.004, **_THRESHOLDS) == []


@pytest.mark.parametrize("scale", [0.915, 0.947, 0.975, 1.024, 1.033, 1.053])
def test_off_grid_scale_is_rejected(scale):
    # These are the scales wrong-episode subtitles got from GSS on real media.
    reasons = assess_alignment_quality(500.0, 3.0, scale, **_THRESHOLDS)
    assert len(reasons) == 1
    assert "not a known framerate ratio" in reasons[0]


def test_tightened_framerate_threshold_rejects_correction():
    reasons = assess_alignment_quality(
        500.0,
        3.0,
        25.0 / 24.0,
        min_score=0.0,
        max_offset_seconds=60.0,
        max_framerate_deviation=0.01,
    )
    assert len(reasons) == 1
    assert "framerate" in reasons[0]


def test_large_deviation_reports_one_framerate_reason():
    reasons = assess_alignment_quality(500.0, 3.0, 1.5, **_THRESHOLDS)
    assert len(reasons) == 1
    assert "deviation" in reasons[0]


def test_quality_reports_multiple_reasons():
    reasons = assess_alignment_quality(
        -5.0,
        99.0,
        1.5,
        min_score=0.0,
        max_offset_seconds=60.0,
        max_framerate_deviation=0.1,
    )
    assert len(reasons) == 3


# ---- end-to-end behavior in try_sync -------------------------------------


def _fake_aligner(score, offset_samples):
    class _Fake:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, refstring, subpipes):
            return (score, offset_samples), subpipes[0]

    return _Fake


def _run_try_sync(tmp_path, monkeypatch, *, score, offset_seconds, **arg_overrides):
    srtin = tmp_path / "in.srt"
    srtin.write_text(SRT)
    srtout = tmp_path / "out.srt"
    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(srtin), "-o", str(srtout)]
    )
    args.skip_infer_framerate_ratio = True
    args.auto_sync = False  # primary strategy only; the fake aligner is fixed
    for k, v in arg_overrides.items():
        setattr(args, k, v)
    monkeypatch.setattr(
        ffsubsync,
        "MaxScoreAligner",
        _fake_aligner(score, int(offset_seconds * SAMPLE_RATE)),
    )
    reference_pipe = types.SimpleNamespace(transform=lambda _: np.zeros(10))
    result = {"retval": 0}
    ok = ffsubsync.try_sync(args, reference_pipe, result)
    return ok, srtout, result


def _first_start_seconds(path):
    parser = GenericSubtitleParser()
    parser.fit(str(path))
    return next(iter(parser.subs_)).start.total_seconds()


def test_try_sync_keeps_original_on_negative_score(tmp_path, monkeypatch):
    ok, srtout, result = _run_try_sync(
        tmp_path,
        monkeypatch,
        score=-3.0,
        offset_seconds=5.0,
        skip_sync_on_low_quality=True,
    )
    assert ok is False
    assert _first_start_seconds(srtout) == pytest.approx(10.0)  # unchanged
    assert "score" in result["kept_original_reason"]


def test_try_sync_keeps_original_on_large_offset(tmp_path, monkeypatch):
    ok, srtout, result = _run_try_sync(
        tmp_path,
        monkeypatch,
        score=500.0,
        offset_seconds=300.0,
        skip_sync_on_low_quality=True,
        max_offset_seconds=600,
    )
    assert ok is False
    assert _first_start_seconds(srtout) == pytest.approx(10.0)
    assert "offset" in result["kept_original_reason"]


def test_try_sync_gate_blocks_piecewise(tmp_path, monkeypatch):
    def fail_if_called(*_a, **_k):
        raise AssertionError("piecewise must not run on a rejected sync")

    monkeypatch.setattr(ffsubsync, "_compute_piecewise_anchors", fail_if_called)
    ok, srtout, _ = _run_try_sync(
        tmp_path,
        monkeypatch,
        score=-3.0,
        offset_seconds=5.0,
        skip_sync_on_low_quality=True,
        piecewise_audio=True,
    )
    assert ok is False
    assert _first_start_seconds(srtout) == pytest.approx(10.0)


def test_try_sync_applies_when_quality_is_acceptable(tmp_path, monkeypatch):
    ok, srtout, result = _run_try_sync(
        tmp_path,
        monkeypatch,
        score=500.0,
        offset_seconds=5.0,
        skip_sync_on_low_quality=True,
    )
    assert ok is True
    assert _first_start_seconds(srtout) == pytest.approx(15.0)  # shifted +5s
    assert "kept_original_reason" not in result


def test_try_sync_ignores_quality_without_flag(tmp_path, monkeypatch):
    # Same bad offset, but the flag is off, so it is applied anyway.
    ok, srtout, _ = _run_try_sync(
        tmp_path,
        monkeypatch,
        score=500.0,
        offset_seconds=300.0,
        skip_sync_on_low_quality=False,
        max_offset_seconds=600,
    )
    assert ok is True
    assert _first_start_seconds(srtout) == pytest.approx(310.0)


def test_cli_parses_quality_flags():
    args = ffsubsync.make_parser().parse_args(
        [
            "movie.mkv",
            "--skip-sync-on-low-quality",
            "--min-score",
            "100",
            "--quality-max-offset-seconds",
            "20",
            "--max-framerate-deviation",
            "0.03",
        ]
    )
    assert args.skip_sync_on_low_quality is True
    assert args.min_score == 100.0
    assert args.quality_max_offset_seconds == 20.0
    assert args.max_framerate_deviation == 0.03


def test_ffs_cli_gate_is_off_by_default():
    args = ffsubsync.make_parser().parse_args(["movie.mkv"])
    assert args.skip_sync_on_low_quality is False


def test_result_keys_do_not_leak_between_input_files(tmp_path, monkeypatch):
    bad = tmp_path / "a.srt"
    good = tmp_path / "b.srt"
    bad.write_text(SRT)
    good.write_text(SRT)
    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(bad), str(good), "--overwrite-input"]
    )
    args.skip_infer_framerate_ratio = True
    args.auto_sync = False
    args.skip_sync_on_low_quality = True
    scores = iter([-3.0, 500.0])

    class _Fake:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, refstring, subpipes):
            return (next(scores), 5 * SAMPLE_RATE), subpipes[0]

    monkeypatch.setattr(ffsubsync, "MaxScoreAligner", _Fake)
    reference_pipe = types.SimpleNamespace(transform=lambda _: np.zeros(10))
    result = {"retval": 0}
    ffsubsync.try_sync(args, reference_pipe, result)

    assert "kept_original_reason" not in result  # b.srt synced fine
    assert _first_start_seconds(bad) == pytest.approx(10.0)
    assert _first_start_seconds(good) == pytest.approx(15.0)
