"""
Fast pre-flight subtitle sync check.

Extracts only the first few minutes of audio and runs FFTAligner on the
short window. If the subtitle is already aligned (offset ≤ 500 ms with
a confident score margin) the caller can skip the expensive full-pipeline.

Abstains (returns False) when:
- The probe window has too little speech (< 5% voiced frames) — silent intros.
- The score margin is insufficient — ambiguous correlation.
- The subtitle has no dialogue cues in the probe window.
"""

import argparse
import logging
import os

import numpy as np

from ffsubsync.aligners import FFTAligner
from ffsubsync.constants import DEFAULT_VAD, SAMPLE_RATE
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    _is_metadata,
    make_subtitle_speech_pipeline,
)

logger: logging.Logger = logging.getLogger(__name__)

_PREFLIGHT_DIALOGUE_CUES = 5
_PREFLIGHT_PADDING_SECONDS = 30.0
_PREFLIGHT_MIN_DURATION_SECONDS = 120.0
_PREFLIGHT_MAX_DURATION_SECONDS = 240.0
_PREFLIGHT_OFFSET_THRESHOLD = 0.5  # seconds
_PREFLIGHT_SCORE_MARGIN = 1.5  # adaptive must be >= this x second-best
_PREFLIGHT_MIN_SPEECH_FRACTION = 0.05


def _probe_duration(srtin: str, fmt: str) -> float | None:
    """Compute probe window duration from the first dialogue cues."""
    try:
        # We need to parse just to read cues — use GenericSubtitleParser directly.
        from ffsubsync.subtitle_parser import make_subtitle_parser

        parser = make_subtitle_parser(fmt=fmt)
        parser.fit(srtin)
        subs = list(parser.subs_)
    except Exception as e:
        logger.debug("preflight: could not parse subtitle for probe duration: %s", e)
        return _PREFLIGHT_MIN_DURATION_SECONDS

    dialogue_end_times = []
    for i, sub in enumerate(subs):
        if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)):
            continue
        dialogue_end_times.append(sub.end.total_seconds())
        if len(dialogue_end_times) >= _PREFLIGHT_DIALOGUE_CUES:
            break

    if not dialogue_end_times:
        return _PREFLIGHT_MIN_DURATION_SECONDS

    last_cue_end = max(dialogue_end_times)
    duration = last_cue_end + _PREFLIGHT_PADDING_SECONDS
    return max(
        _PREFLIGHT_MIN_DURATION_SECONDS, min(duration, _PREFLIGHT_MAX_DURATION_SECONDS)
    )


def check_already_synced(
    video: str,
    srtin: str,
    args: argparse.Namespace,
    offset_threshold: float = _PREFLIGHT_OFFSET_THRESHOLD,
    score_margin: float = _PREFLIGHT_SCORE_MARGIN,
    min_speech_fraction: float = _PREFLIGHT_MIN_SPEECH_FRACTION,
) -> tuple[bool, float | None]:
    """
    Return (already_synced, offset_seconds).

    Returns (True, offset_seconds) when the subtitle appears aligned in the
    probe window. Returns (False, None) on abstention or detected mis-sync.
    """
    srtin_fmt = "srt" if srtin is None else os.path.splitext(srtin)[-1][1:]
    probe_duration = _probe_duration(srtin, srtin_fmt)
    if probe_duration is None:
        return False, None

    logger.info(
        "preflight: checking first %.0fs of audio (srt=%s)",
        probe_duration,
        os.path.basename(srtin) if srtin else "stdin",
    )

    vad = args.vad or DEFAULT_VAD
    if "subs" in vad:
        vad = "webrtc"

    try:
        ref_transformer = VideoSpeechTransformer(
            vad=vad,
            sample_rate=SAMPLE_RATE,
            frame_rate=args.frame_rate,
            non_speech_label=args.non_speech_label,
            start_seconds=args.start_seconds,
            ffmpeg_path=args.ffmpeg_path,
            ref_stream=None,
            vlc_mode=True,  # suppress tqdm output during preflight
            vad_smoothing_window=args.vad_smoothing_window,
            max_duration_seconds=probe_duration,
        )
        ref_transformer.fit(video)
        ref_speech = ref_transformer.transform(video)
    except Exception as e:
        logger.debug("preflight: audio extraction failed: %s", e)
        return False, None

    # Check minimum speech presence in the probe window.
    speech_fraction = float(np.mean(ref_speech > 0))
    if speech_fraction < min_speech_fraction:
        logger.info(
            "preflight: abstaining — probe window has only %.1f%% speech "
            "(< %.0f%% minimum); likely silent intro",
            speech_fraction * 100,
            min_speech_fraction * 100,
        )
        return False, None

    # Build subtitle speech for the probe window.
    try:
        srt_pipe = make_subtitle_speech_pipeline(
            fmt=srtin_fmt,
            caching=False,
            scale_factor=1.0,
            non_speech_label=args.non_speech_label,
            start_seconds=args.start_seconds,
        )
        sub_speech = srt_pipe.fit_transform(srtin)
    except Exception as e:
        logger.debug("preflight: subtitle speech extraction failed: %s", e)
        return False, None

    # Truncate sub_speech to match ref_speech length.
    n = len(ref_speech)
    if len(sub_speech) < n:
        sub_speech = np.pad(sub_speech, (0, n - len(sub_speech)))
    else:
        sub_speech = sub_speech[:n]

    # Run FFTAligner on the short vectors.
    try:
        max_offset_samples = int(offset_threshold * SAMPLE_RATE * 2)
        aligner = FFTAligner(max_offset_samples=max_offset_samples)
        aligner.fit(ref_speech, sub_speech, get_score=True)
        best_score = aligner.best_score_
        best_offset = aligner.best_offset_
    except Exception as e:
        logger.debug("preflight: FFTAligner failed: %s", e)
        return False, None

    if best_score is None or best_offset is None:
        return False, None

    offset_seconds = best_offset / float(SAMPLE_RATE)

    # Score margin check: also run without offset cap to get second-best context.
    aligner_full = FFTAligner(max_offset_samples=None)
    aligner_full.fit(ref_speech, sub_speech, get_score=True)
    full_best_score = aligner_full.best_score_ or 0.0

    # Guard: if the constrained and unconstrained peaks are nearly equal,
    # the near-zero offset is genuinely dominant. Otherwise the true peak
    # is elsewhere — abstain.
    if full_best_score > 0 and best_score < full_best_score / score_margin:
        logger.info(
            "preflight: abstaining — score at near-zero offset (%.0f) is less than "
            "%.1fx the global best (%.0f); subtitle is likely out of sync",
            best_score,
            score_margin,
            full_best_score,
        )
        return False, None

    if abs(offset_seconds) > offset_threshold:
        logger.info(
            "preflight: detected offset %.3fs exceeds threshold %.3fs; "
            "subtitle needs sync",
            offset_seconds,
            offset_threshold,
        )
        return False, None

    logger.info(
        "preflight: subtitle appears in sync (offset=%.3fs, score=%.0f); "
        "skipping full pipeline",
        offset_seconds,
        best_score,
    )
    return True, offset_seconds
