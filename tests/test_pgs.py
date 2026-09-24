"""Tests for PGS (image-based subtitle) timings as a sync reference.

Ported from upstream tests/test_pgs.py. The fork also recovers caption ends from
the next packet, because some muxers store no duration on PGS packets.
"""

from unittest.mock import patch

import numpy as np
import pytest

from ffsubsync import ffsubsync
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.speech_transformers import (
    PGSSpeechTransformer,
    _get_pgs_timings_via_ffprobe,
    find_pgs_stream,
)

PROBE = "ffsubsync.speech_transformers.ffmpeg.probe"
BIN = "ffsubsync.speech_transformers.ffmpeg_bin_path"


def _packet(pts_time, duration_time, size):
    packet = {"pts_time": str(pts_time), "size": str(size)}
    if duration_time is not None:
        packet["duration_time"] = str(duration_time)
    return packet


def _timings(packets):
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.return_value = {"packets": packets}
        return _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")


def test_uses_duration_when_present():
    result = _timings([_packet(1.0, 2.5, 1000), _packet(5.0, 1.0, 800)])
    assert result == [(1.0, 3.5), (5.0, 6.0)]


def test_caption_ends_at_clear_packet_when_duration_is_missing():
    # This is how a real Blu-ray remux looks: no duration_time key at all.
    result = _timings(
        [
            _packet(27.028, None, 4541),
            _packet(28.029, None, 30),
            _packet(33.326, None, 16789),
            _packet(35.077, None, 30),
        ]
    )
    assert result == [(27.028, 28.029), (33.326, 35.077)]


def test_na_duration_is_treated_as_missing():
    result = _timings([_packet(1.0, "N/A", 1000), _packet(2.0, "N/A", 30)])
    assert result == [(1.0, 2.0)]


def test_caption_without_clear_packet_is_capped():
    # A lost clear event must not turn one caption into minutes of speech.
    result = _timings([_packet(1.0, None, 1000), _packet(100.0, None, 900)])
    assert result is not None
    assert result[0] == (1.0, 11.0)


def test_last_packet_without_duration_is_dropped():
    assert _timings([_packet(1.0, None, 1000)]) is None


def test_skips_clear_events_small_size():
    result = _timings([_packet(1.0, 2.0, 1000), _packet(3.0, 0.001, 30)])
    assert result == [(1.0, 3.0)]


def test_packets_are_sorted_by_time():
    result = _timings(
        [_packet(5.0, None, 30), _packet(3.0, None, 900), _packet(1.0, 1.0, 900)]
    )
    assert result == [(1.0, 2.0), (3.0, 5.0)]


def test_skips_packets_with_missing_fields():
    result = _timings(
        [
            {"pts_time": "1.0", "duration_time": "2.0"},  # missing size
            {"duration_time": "1.0", "size": "500"},  # missing pts_time
            _packet(10.0, 1.0, 200),
        ]
    )
    assert result == [(10.0, 11.0)]


def test_returns_none_on_empty_packets():
    assert _timings([]) is None


def test_returns_none_when_ffprobe_raises():
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.side_effect = Exception("ffprobe not found")
        assert _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0") is None


@pytest.mark.parametrize("stream, expected", [("0:s:0", "s:0"), ("s:1", "s:1")])
def test_strips_input_prefix_for_ffprobe(stream, expected):
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.return_value = {"packets": [_packet(0.0, 1.0, 100)]}
        _get_pgs_timings_via_ffprobe("test.mkv", stream)
        assert probe.call_args.kwargs["select_streams"] == expected


def test_find_pgs_stream_counts_subtitle_streams_only():
    streams = [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "dts"},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip"},
        {"index": 3, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
    ]
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.return_value = {"streams": streams}
        assert find_pgs_stream("movie.mkv") == "0:s:1"


def test_find_pgs_stream_none_without_pgs():
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.return_value = {"streams": [{"codec_type": "subtitle"}]}
        assert find_pgs_stream("movie.mkv") is None


def test_transformer_builds_binary_speech_signal():
    with patch(
        "ffsubsync.speech_transformers._get_pgs_timings_via_ffprobe",
        return_value=[(1.0, 2.0), (3.0, 3.5)],
    ):
        transformer = PGSSpeechTransformer(SAMPLE_RATE, ref_stream="s:0").fit("m.mkv")
    speech = transformer.transform()
    assert np.all(speech[100:200] == 1.0)
    assert np.all(speech[200:300] == 0.0)
    assert np.all(speech[300:350] == 1.0)
    assert speech[:100].sum() == 0
    # The signal is already in the video timebase: no length-based framerate guess.
    assert transformer.num_frames is None


def test_transformer_raises_without_timings():
    with (
        patch(
            "ffsubsync.speech_transformers._get_pgs_timings_via_ffprobe",
            return_value=None,
        ),
        pytest.raises(ValueError, match="No usable PGS caption timings"),
    ):
        PGSSpeechTransformer(SAMPLE_RATE, ref_stream="0:s:0").fit("m.mkv")


def _parse(argv):
    return ffsubsync.make_parser().parse_args(argv)


def test_cli_bare_flag_means_auto():
    assert _parse(["movie.mkv", "--pgs-ref-stream"]).pgs_ref_stream == "auto"
    assert _parse(["movie.mkv", "--pgs-ref-stream", "s:2"]).pgs_ref_stream == "s:2"
    assert _parse(["movie.mkv"]).pgs_ref_stream is None


def test_reference_pipe_uses_pgs_transformer():
    args = _parse(["movie.mkv", "--pgs-ref-stream", "s:2"])
    pipe = ffsubsync.make_reference_pipe(args)
    extractor = pipe.named_steps["speech_extract"]
    assert isinstance(extractor, PGSSpeechTransformer)
    assert extractor.ref_stream == "s:2"


def test_reference_pipe_auto_detects_when_flag_is_bare():
    pipe = ffsubsync.make_reference_pipe(_parse(["movie.mkv", "--pgs-ref-stream"]))
    assert pipe.named_steps["speech_extract"].ref_stream is None


def test_pgs_flag_rejects_non_video_reference():
    args = _parse(["ref.srt", "-i", "in.srt", "--pgs-ref-stream"])
    with pytest.raises(ValueError, match="needs a video"):
        ffsubsync.validate_args(args)


def test_container_start_time_is_subtracted():
    # MPEG-TS starts at a non-zero timestamp; every other reference counts from it.
    with patch(BIN, return_value="ffprobe"), patch(PROBE) as probe:
        probe.return_value = {
            "format": {"start_time": "1.400000"},
            "packets": [_packet(11.4, None, 900), _packet(12.4, None, 30)],
        }
        result = _get_pgs_timings_via_ffprobe("test.ts", "0:s:0")
    assert result is not None
    assert result[0] == pytest.approx((10.0, 11.0))


def test_try_sync_with_pgs_reference_end_to_end(tmp_path):
    # Regression: PGSSpeechTransformer.num_frames is None, and compute_alignment
    # used to call float(None) on it, so every strategy failed.
    srtin = tmp_path / "in.srt"
    srtin.write_text(
        "1\n00:00:05,000 --> 00:00:06,000\nhi\n\n"
        "2\n00:00:09,000 --> 00:00:10,500\nyo\n\n"
        "3\n00:00:15,000 --> 00:00:15,700\nok\n"
    )
    srtout = tmp_path / "out.srt"
    args = _parse(["m.mkv", "-i", str(srtin), "-o", str(srtout), "--pgs-ref-stream"])
    with (
        patch("ffsubsync.speech_transformers.find_pgs_stream", return_value="0:s:0"),
        patch(
            "ffsubsync.speech_transformers._get_pgs_timings_via_ffprobe",
            return_value=[(3.0, 4.0), (7.0, 8.5), (13.0, 13.7)],
        ),
    ):
        pipe = ffsubsync.make_reference_pipe(args).fit("m.mkv")
    result = {}

    assert ffsubsync.try_sync(args, pipe, result) is True
    assert result["offset_seconds"] == pytest.approx(-2.0)
