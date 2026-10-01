import io
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from datetime import timedelta
from typing import cast

import ffmpeg
import numpy as np
import tqdm

from ffsubsync.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_START_SECONDS,
    SAMPLE_RATE,
    VAD_FRAME_RATE,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleScaler

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def make_subtitle_speech_pipeline(
    fmt: str = "srt",
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    parser=None,
    **kwargs,
) -> Pipeline | Callable[[float], Pipeline]:
    if parser is None:
        parser = make_subtitle_parser(
            fmt,
            encoding=encoding,
            caching=caching,
            max_subtitle_seconds=max_subtitle_seconds,
            start_seconds=start_seconds,
            **kwargs,
        )
    assert parser.encoding == encoding
    assert parser.max_subtitle_seconds == max_subtitle_seconds
    assert parser.start_seconds == start_seconds

    def subpipe_maker(framerate_ratio):
        return Pipeline(
            [
                ("parse", parser),
                ("scale", SubtitleScaler(framerate_ratio)),
                (
                    "speech_extract",
                    SubtitleSpeechTransformer(
                        sample_rate=SAMPLE_RATE,
                        start_seconds=start_seconds,
                        framerate_ratio=framerate_ratio,
                    ),
                ),
            ]
        )

    if scale_factor is None:
        return subpipe_maker
    else:
        return subpipe_maker(scale_factor)


def _smooth_speech(raw_speech: np.ndarray, window_size: int = 30) -> np.ndarray:
    """
    Fills gaps in speech using a simple convolution (dilation).

    window_size: Number of frames for gap filling.
                 Default 30 frames @ 100Hz = 300ms gap filling.

    Uses numpy convolution to avoid scipy dependency.
    """
    if window_size <= 1:
        return raw_speech

    # Create a kernel of ones (this acts as the dilation window)
    kernel = np.ones(window_size)

    # Convolve. If any speech frame falls within the window, the result > 0
    smoothed = np.convolve(raw_speech, kernel, mode="same")

    # Convert back to binary (0.0 or 1.0)
    return (smoothed > 0).astype(float)


def _make_webrtcvad_detector(
    sample_rate: int,
    frame_rate: int,
    smoothing_window_size: int = 30,
) -> Callable[[bytes], np.ndarray]:
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 2
    chunk_size = frames_per_window * bytes_per_frame

    def _detect(asegment: bytes) -> np.ndarray:
        # Create a zero-copy memory view to avoid millions of allocations
        mem_view = memoryview(asegment)
        media_bstring = []
        n_bytes = len(asegment)

        # Iterate using pre-calculated chunk size
        for start in range(0, n_bytes, chunk_size):
            # Ensure we don't go out of bounds (drop incomplete last chunk)
            if start + chunk_size > n_bytes:
                break

            # Zero-copy slice
            chunk = mem_view[start : start + chunk_size]

            try:
                # bytes(chunk) is needed by webrtcvad C-extension
                # but we saved the slice creation overhead
                is_speech = vad.is_speech(bytes(chunk), sample_rate=frame_rate)
            except Exception:
                is_speech = False
            media_bstring.append(1.0 if is_speech else 0.0)

        result = np.array(media_bstring)

        # Fill gaps in speech using smoothing window
        # This matches subtitle characteristics better than raw VAD
        return _smooth_speech(result, window_size=smoothing_window_size)

    return _detect


class ComputeSpeechFrameBoundariesMixin:
    def __init__(self) -> None:
        self.start_frame_: int | None = None
        self.end_frame_: int | None = None

    @property
    def num_frames(self) -> int | None:
        if self.start_frame_ is None or self.end_frame_ is None:
            return None
        return self.end_frame_ - self.start_frame_

    def fit_boundaries(
        self, speech_frames: np.ndarray
    ) -> "ComputeSpeechFrameBoundariesMixin":
        nz = np.nonzero(speech_frames > 0.5)[0]
        if len(nz) > 0:
            self.start_frame_ = int(np.min(nz))
            self.end_frame_ = int(np.max(nz))
        return self


class VideoSpeechTransformer(TransformerMixin):
    def __init__(
        self,
        vad: str,
        sample_rate: int,
        start_seconds: int = 0,
        ffmpeg_path: str | None = None,
        ref_stream: str | None = None,
        vlc_mode: bool = False,
        vad_smoothing_window: int = 30,
        max_duration_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.vad: str = vad
        self.sample_rate: int = sample_rate
        self.vad_smoothing_window: int = vad_smoothing_window
        self.max_duration_seconds: float | None = max_duration_seconds
        self.frame_rate: int = VAD_FRAME_RATE
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: str | None = ffmpeg_path
        self.ref_stream: str | None = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.video_speech_results_: np.ndarray | None = None

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        embedded_subs = []
        embedded_subs_times = []
        if self.ref_stream is None:
            # check first 5; should cover 99% of movies
            streams_to_try: list[str] = list(map("0:s:{}".format, range(5)))
        else:
            streams_to_try = [self.ref_stream]
        for stream in streams_to_try:
            ffmpeg_args = [
                ffmpeg_bin_path("ffmpeg", ffmpeg_resources_path=self.ffmpeg_path)
            ]
            ffmpeg_args.extend(
                [
                    "-loglevel",
                    "fatal",
                    "-nostdin",
                    "-i",
                    fname,
                    "-map",
                    f"{stream}",
                    "-f",
                    "srt",
                    "-",
                ]
            )
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            output = io.BytesIO(process.communicate()[0])
            if process.returncode != 0:
                break
            pipe = cast(
                Pipeline,
                make_subtitle_speech_pipeline(start_seconds=self.start_seconds),
            ).fit(output)
            speech_step = pipe.steps[-1][1]
            embedded_subs.append(speech_step)
            embedded_subs_times.append(speech_step.max_time_)
        if len(embedded_subs) == 0:
            if self.ref_stream is None:
                error_msg = "Video file appears to lack subtitle stream"
            else:
                error_msg = f"Stream {self.ref_stream} not found"
            raise ValueError(error_msg)
        # use longest set of embedded subs
        subs_to_use = embedded_subs[int(np.argmax(embedded_subs_times))]
        self.video_speech_results_ = subs_to_use.subtitle_speech_results_

    def fit(self, fname: str, *_) -> "VideoSpeechTransformer":
        if "subs" in self.vad and (
            self.ref_stream is None or self.ref_stream.startswith("0:s:")
        ):
            try:
                logger.info("Checking video for subtitles stream...")
                self.try_fit_using_embedded_subs(fname)
                logger.info("...success!")
                return self
            except Exception as e:
                logger.info(e)
        try:
            total_duration = (
                float(
                    ffmpeg.probe(
                        fname,
                        cmd=ffmpeg_bin_path(
                            "ffprobe",
                            ffmpeg_resources_path=self.ffmpeg_path,
                        ),
                    )["format"]["duration"]
                )
                - self.start_seconds
            )
        except Exception as e:
            logger.warning(e)
            total_duration = None
        detector = _make_webrtcvad_detector(
            self.sample_rate, self.frame_rate, self.vad_smoothing_window
        )
        media_bstring: list[np.ndarray] = []
        ffmpeg_args = [
            ffmpeg_bin_path("ffmpeg", ffmpeg_resources_path=self.ffmpeg_path)
        ]
        if self.start_seconds > 0:
            ffmpeg_args.extend(
                [
                    "-ss",
                    str(timedelta(seconds=self.start_seconds)),
                ]
            )
        ffmpeg_args.extend(["-loglevel", "fatal", "-nostdin", "-i", fname])
        if self.ref_stream is not None and self.ref_stream.startswith("0:a:"):
            ffmpeg_args.extend(["-map", self.ref_stream])
        if self.max_duration_seconds is not None:
            ffmpeg_args.extend(["-t", str(self.max_duration_seconds)])
            if total_duration is not None:
                total_duration = min(total_duration, self.max_duration_seconds)
        ffmpeg_args.extend(
            [
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-af",
                "aresample=async=1",
                "-ar",
                str(self.frame_rate),
                "-",
            ]
        )
        process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
        bytes_per_frame = 2
        frames_per_window = bytes_per_frame * self.frame_rate // self.sample_rate
        windows_per_buffer = 10000
        simple_progress = 0.0

        redirect_stderr = None
        tqdm_extra_args = {}
        if redirect_stderr is None:

            @contextmanager
            def redirect_stderr(enter_result=None):
                yield enter_result

        assert redirect_stderr is not None
        pbar_output = io.StringIO()
        with (
            redirect_stderr(pbar_output),
            tqdm.tqdm(
                total=total_duration, disable=self.vlc_mode, **tqdm_extra_args
            ) as pbar,
        ):
            while True:
                in_bytes = process.stdout.read(frames_per_window * windows_per_buffer)
                if not in_bytes:
                    break
                newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                if (
                    total_duration is not None
                    and simple_progress + newstuff > total_duration
                ):
                    newstuff = total_duration - simple_progress
                simple_progress += newstuff
                pbar.update(newstuff)
                if self.vlc_mode and total_duration is not None:
                    print(f"{int(simple_progress * 100.0 / total_duration)}")
                    sys.stdout.flush()

                in_bytes = np.frombuffer(in_bytes, np.uint8)
                media_bstring.append(detector(in_bytes))
        process.wait()
        if len(media_bstring) == 0:
            raise ValueError(
                "Unable to detect speech. "
                "Perhaps try specifying a different stream / track."
            )
        self.video_speech_results_ = np.concatenate(media_bstring)
        logger.info("total of speech segments: %s", np.sum(self.video_speech_results_))
        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


_PAIRED_NESTER: dict[str, str] = {
    "(": ")",
    "{": "}",
    "[": "]",
    "（": "）",  # noqa: RUF001  full-width / CJK brackets, common outside English
    "【": "】",
    # Not 「」: in Japanese it is the ordinary quotation mark around dialogue.
}

# Markup tags (e.g. <i>, </i>, <font ...>) carry no speech. Stripping them before
# classifying a line recognizes a wrapped cue like "<i>[music]</i>" as non-dialogue
# while "<i>Hello?</i>" stays dialogue. That is why '<' is not a paired nester.
_MARKUP_TAG: re.Pattern[str] = re.compile(r"<[^>]+>")

# Symbols that, on their own, denote a musical / non-speech cue.
_NON_DIALOGUE_SYMBOLS: frozenset[str] = frozenset("♪♫♬♩🎵🎶")


# TODO: need way better metadata detector
def _is_metadata(content: str, is_beginning_or_end: bool) -> bool:
    content = _MARKUP_TAG.sub("", content).strip()
    if len(content) == 0:
        return True
    if content[0] in _PAIRED_NESTER and content[-1] == _PAIRED_NESTER[content[0]]:
        return True
    # lines consisting only of music notes / sound symbols are cues, not speech
    if all(ch.isspace() or ch in _NON_DIALOGUE_SYMBOLS for ch in content):
        return True
    if is_beginning_or_end:
        if "english" in content.lower():
            return True
        if " - " in content:
            return True
    return False


class SubtitleSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    def __init__(
        self, sample_rate: int, start_seconds: int = 0, framerate_ratio: float = 1.0
    ) -> None:
        super().__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.framerate_ratio: float = framerate_ratio
        self.subtitle_speech_results_: np.ndarray | None = None
        self.max_time_: int | None = None

    def fit(self, subs: list[GenericSubtitle], *_) -> "SubtitleSpeechTransformer":
        max_time = 0
        for sub in subs:
            max_time = max(max_time, sub.end.total_seconds())
        self.max_time_ = max_time - self.start_seconds
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=float)
        start_frame = float("inf")
        end_frame = 0
        for i, sub in enumerate(subs):
            if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)):
                continue
            start = round(
                (sub.start.total_seconds() - self.start_seconds) * self.sample_rate
            )
            start_frame = min(start_frame, start)
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + round(duration * self.sample_rate)
            end_frame = max(end_frame, end)
            samples[start:end] = min(1.0 / self.framerate_ratio, 1.0)
        self.subtitle_speech_results_ = samples
        self.fit_boundaries(self.subtitle_speech_results_)
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.subtitle_speech_results_ is not None
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.deserialized_speech_results_: np.ndarray | None = None

    def fit(self, fname, *_) -> "DeserializeSpeechTransformer":
        speech = np.load(fname)
        if hasattr(speech, "files"):
            if "speech" in speech.files:
                speech = speech["speech"]
            else:
                raise ValueError(
                    'could not find "speech" array in '
                    f"serialized file; only contains: {speech.files}"
                )
        speech[speech < 1.0] = 0.0
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.deserialized_speech_results_ is not None
        return self.deserialized_speech_results_


PGS_CODEC: str = "hdmv_pgs_subtitle"
# PGS "clear" packets remove the caption from the screen and carry no image. They
# are about 30 bytes, so anything this small is not a displayed caption.
_PGS_MIN_SHOW_PACKET_BYTES: int = 50
# Upper bound for a caption whose end is taken from the next packet, so a missing
# clear event cannot turn one caption into minutes of "speech".
_PGS_MAX_CAPTION_SECONDS: float = float(DEFAULT_MAX_SUBTITLE_SECONDS)


def find_pgs_stream(fname: str, ffmpeg_path: str | None = None) -> str | None:
    """Return the ffmpeg specifier (e.g. ``"0:s:1"``) of the first PGS track.

    Returns ``None`` when ffprobe fails or the file has no PGS subtitle stream.
    """
    try:
        probe = ffmpeg.probe(
            fname, cmd=ffmpeg_bin_path("ffprobe", ffmpeg_resources_path=ffmpeg_path)
        )
    except Exception as e:
        logger.warning("ffprobe failed while searching for PGS streams: %s", e)
        return None

    sub_index = 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        if stream.get("codec_name") == PGS_CODEC:
            specifier = f"0:s:{sub_index}"
            logger.info(
                "auto-detected PGS stream: %s (ffmpeg stream index %s)",
                specifier,
                stream.get("index"),
            )
            return specifier
        sub_index += 1
    return None


def _get_pgs_timings_via_ffprobe(
    fname: str, stream: str, ffmpeg_path: str | None = None
) -> list[tuple[float, float]] | None:
    """Read PGS caption timings from container packet metadata with ffprobe.

    The container stores a presentation timestamp for every subtitle packet, so
    caption times are available without decoding the bitmaps. A caption starts
    at a large "show" packet and ends at the next tiny "clear" packet. Some
    muxers also store ``duration_time`` on show packets; it is used when present,
    otherwise the caption ends at the next packet, capped at
    ``_PGS_MAX_CAPTION_SECONDS``.

    Returns ``(start_seconds, end_seconds)`` tuples, or ``None`` when ffprobe
    fails or no caption can be recovered.
    """
    # ffprobe -select_streams does not accept the "0:" input-index prefix.
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    try:
        probe_data = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", ffmpeg_resources_path=ffmpeg_path),
            show_packets=None,
            select_streams=probe_stream,
            show_entries="packet=pts_time,duration_time,size:format=start_time",
        )
    except Exception:
        return None

    # pts_time is the raw container timestamp. ffmpeg's audio extraction (and so
    # every other reference) counts from the container start time, which is
    # about 0 for MKV but not for MPEG-TS, so subtract it.
    try:
        container_start = float(probe_data.get("format", {}).get("start_time", 0.0))
    except (TypeError, ValueError):
        container_start = 0.0

    packets: list[tuple[float, float | None, int]] = []
    for packet in probe_data.get("packets", []):
        try:
            pts_time = float(packet["pts_time"]) - container_start
            size = int(packet["size"])
        except (KeyError, TypeError, ValueError):
            continue
        duration: float | None
        try:
            duration = float(packet.get("duration_time", "N/A"))
        except (TypeError, ValueError):
            duration = None
        packets.append((pts_time, duration, size))
    packets.sort(key=lambda p: p[0])

    results: list[tuple[float, float]] = []
    for i, (pts_time, duration, size) in enumerate(packets):
        if size <= _PGS_MIN_SHOW_PACKET_BYTES:
            continue  # a clear event, not a displayed caption
        if duration is not None and duration > 0:
            end = pts_time + duration
        elif i + 1 < len(packets):
            end = min(packets[i + 1][0], pts_time + _PGS_MAX_CAPTION_SECONDS)
        else:
            continue  # last packet with no duration: its end is unknown
        if end > pts_time:
            results.append((pts_time, end))
    return results or None


class PGSSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    """Use the timings of an image-based (PGS, Blu-ray) subtitle track as reference.

    ffmpeg cannot convert PGS to text, but the container stores when each caption
    is on screen. This transformer reads those timings with ffprobe and builds the
    same binary speech signal that :class:`SubtitleSpeechTransformer` builds for
    text subtitles: 1.0 while a caption is displayed, 0.0 otherwise.

    ``ref_stream`` is an ffmpeg stream specifier (with or without a leading
    ``0:``), or ``None`` to auto-detect the first PGS track.
    """

    # PGS timings are already in the video's timebase, so their extent says
    # nothing about a framerate mismatch. None disables the length-based
    # framerate inference in compute_alignment.
    @property
    def num_frames(self) -> None:
        return None

    def __init__(
        self,
        sample_rate: int,
        start_seconds: int = 0,
        ffmpeg_path: str | None = None,
        ref_stream: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: str | None = ffmpeg_path
        self.ref_stream: str | None = ref_stream
        self.pgs_speech_results_: np.ndarray | None = None

    def fit(self, fname: str, *_) -> "PGSSpeechTransformer":
        if self.ref_stream is None:
            stream = find_pgs_stream(fname, self.ffmpeg_path)
            if stream is None:
                raise ValueError(
                    f"No {PGS_CODEC} stream found in {fname}. "
                    "Specify one explicitly with --pgs-ref-stream."
                )
        else:
            stream = self.ref_stream
            if not stream.startswith("0:"):
                stream = "0:" + stream

        logger.info("reading PGS timings for stream %s from %s...", stream, fname)
        timings = _get_pgs_timings_via_ffprobe(fname, stream, self.ffmpeg_path)
        if timings is None:
            raise ValueError(
                f"No usable PGS caption timings in stream {stream} of {fname}. "
                f"Check that it is a {PGS_CODEC} track "
                f"(ffprobe -show_streams {fname})."
            )

        logger.info("found %d PGS subtitle segments", len(timings))
        for i, (start, end) in enumerate(timings[:8]):
            logger.debug(
                "  PGS[%d]: %s --> %s (%.3fs)",
                i,
                timedelta(seconds=start),
                timedelta(seconds=end),
                end - start,
            )

        max_time = max(end for _, end in timings)
        num_samples = int(max_time * self.sample_rate) + 2
        samples = np.zeros(num_samples, dtype=float)
        for start, end in timings:
            start_sample = round((start - self.start_seconds) * self.sample_rate)
            end_sample = round((end - self.start_seconds) * self.sample_rate)
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, num_samples)
            if start_sample < end_sample:
                samples[start_sample:end_sample] = 1.0

        self.pgs_speech_results_ = samples
        self.fit_boundaries(samples)
        logger.info("total PGS subtitle frames: %d", int(np.sum(samples)))
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.pgs_speech_results_ is not None
        return self.pgs_speech_results_
