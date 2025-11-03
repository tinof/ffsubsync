import sys
import types

import numpy as np

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    _make_tenvad_detector,
)
from ffsubsync.ffsubsync import make_parser


def test_tenvad_cli_choice_parses():
    parser = make_parser()
    args = parser.parse_args(["--vad", "tenvad"])  # type: ignore[arg-type]
    assert args.vad == "tenvad"


def test_tenvad_forces_16k_frame_rate():
    vst = VideoSpeechTransformer(
        vad="tenvad",
        sample_rate=SAMPLE_RATE,
        frame_rate=48000,
        non_speech_label=0.0,
    )
    assert vst.frame_rate == 16000


def test_tenvad_detector_mock(monkeypatch):
    # Create a fake ten_vad module with a TenVad class
    fake_mod = types.ModuleType("ten_vad")

    class FakeTenVad:
        def __init__(self, hop_size, threshold):  # noqa: D401 - simple stub
            self.hop_size = hop_size
            self.threshold = threshold

        def process(self, chunk: np.ndarray):
            # Return fixed probability and flag regardless of input
            return 0.8, 1

    fake_mod.TenVad = FakeTenVad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ten_vad", fake_mod)

    # TEN VAD path: 10 ms windows -> hop ~160 samples at 16 kHz
    detector = _make_tenvad_detector(sample_rate=SAMPLE_RATE, frame_rate=16000, non_speech_label=0.0)

    # Build 5 windows of int16 zeros → expect 5 outputs near 0.8
    frames_per_window = int((1.0 / SAMPLE_RATE) * 16000 + 0.5)
    samples = np.zeros(frames_per_window * 5, dtype=np.int16)
    out = detector(samples.tobytes())
    assert len(out) == 5
    # Should pass through the mocked 0.8 probability
    assert np.allclose(out, 0.8, atol=1e-6)


def test_tenvad_fallback_to_webrtc(monkeypatch):
    def raise_import(*_args, **_kwargs):
        raise ImportError('ten-vad missing')

    monkeypatch.setattr(
        'ffsubsync.speech_transformers._make_tenvad_detector', raise_import, raising=False
    )

    called = {}

    def fake_webrtc(sample_rate, frame_rate, non_speech_label):
        called['args'] = (sample_rate, frame_rate, non_speech_label)
        return lambda data: np.zeros(1, dtype=float)

    monkeypatch.setattr(
        'ffsubsync.speech_transformers._make_webrtcvad_detector',
        fake_webrtc,
        raising=False,
    )

    vst = VideoSpeechTransformer(
        vad='tenvad',
        sample_rate=SAMPLE_RATE,
        frame_rate=16000,
        non_speech_label=0.25,
    )

    detector = vst._build_detector()
    assert 'args' in called
    out = detector(b'\x00\x00')
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 1
