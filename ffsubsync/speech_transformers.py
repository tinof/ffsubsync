# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager
import logging
import io
import subprocess
import sys
from datetime import timedelta
from typing import cast, Callable, Dict, List, Optional, Union

import ffmpeg
import numpy as np
import tqdm

from ffsubsync.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_START_SECONDS,
    SAMPLE_RATE,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.sklearn_shim import TransformerMixin
from ffsubsync.sklearn_shim import Pipeline
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
) -> Union[Pipeline, Callable[[float], Pipeline]]:
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


def _make_auditok_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    try:
        from auditok import (
            BufferAudioSource,
            ADSFactory,
            AudioEnergyValidator,
            StreamTokenizer,
        )
    except ImportError as e:
        logger.error(
            """Error: auditok not installed!
        Consider installing it with `pip install auditok`. Note that auditok
        is GPLv3 licensed, which means that successfully importing it at
        runtime creates a derivative work that is GPLv3 licensed. For personal
        use this is fine, but note that any commercial use that relies on
        auditok must be open source as per the GPLv3!*
        *Not legal advice. Consult with a lawyer.
        """
        )
        raise e
    bytes_per_frame = 2
    frames_per_window = frame_rate // sample_rate
    validator = AudioEnergyValidator(sample_width=bytes_per_frame, energy_threshold=50)
    tokenizer = StreamTokenizer(
        validator=validator,
        min_length=0.2 * sample_rate,
        max_length=int(5 * sample_rate),
        max_continuous_silence=0.25 * sample_rate,
    )

    def _detect(asegment: bytes) -> np.ndarray:
        asource = BufferAudioSource(
            data_buffer=asegment,
            sampling_rate=frame_rate,
            sample_width=bytes_per_frame,
            channels=1,
        )
        ads = ADSFactory.ads(audio_source=asource, block_dur=1.0 / sample_rate)
        ads.open()
        tokens = tokenizer.tokenize(ads)
        length = (
            len(asegment) // bytes_per_frame + frames_per_window - 1
        ) // frames_per_window
        media_bstring = np.zeros(length + 1)
        for token in tokens:
            media_bstring[token[1]] = 1.0
            media_bstring[token[2] + 1] = non_speech_label - 1.0
        return np.clip(np.cumsum(media_bstring)[:-1], 0.0, 1.0)

    return _detect


def _make_webrtcvad_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 2

    def _detect(asegment: bytes) -> np.ndarray:
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment) // bytes_per_frame)
            try:
                is_speech = vad.is_speech(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    sample_rate=frame_rate,
                )
            except Exception:
                is_speech = False
                failures += 1
            # webrtcvad has low recall on mode 3, so treat non-speech as "not sure"
            media_bstring.append(1.0 if is_speech else non_speech_label)
        return np.array(media_bstring)

    return _detect


def _make_silero_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    import torch

    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 1

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    exception_logged = False

    def _detect(asegment) -> np.ndarray:
        asegment = np.frombuffer(asegment, np.int16).astype(np.float32) / (1 << 15)
        asegment = torch.FloatTensor(asegment)
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment))
            try:
                speech_prob = model(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    frame_rate,
                ).item()
            except Exception:
                nonlocal exception_logged
                if not exception_logged:
                    exception_logged = True
                    logger.exception("exception occurred during speech detection")
                speech_prob = 0.0
                failures += 1
            media_bstring.append(1.0 - (1.0 - speech_prob) * (1.0 - non_speech_label))
        return np.array(media_bstring)

    return _detect


class ComputeSpeechFrameBoundariesMixin:
    def __init__(self) -> None:
        self.start_frame_: Optional[int] = None
        self.end_frame_: Optional[int] = None

    @property
    def num_frames(self) -> Optional[int]:
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
        frame_rate: int,
        non_speech_label: float,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
    ) -> None:
        super(VideoSpeechTransformer, self).__init__()
        self.vad: str = vad
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.video_speech_results_: Optional[np.ndarray] = None

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        embedded_subs = []
        embedded_subs_times = []
        if self.ref_stream is None:
            # check first 5; should cover 99% of movies
            streams_to_try: List[str] = list(map("0:s:{}".format, range(5)))
        else:
            streams_to_try = [self.ref_stream]
        for stream in streams_to_try:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", ffmpeg_resources_path=self.ffmpeg_path
                )
            ]
            ffmpeg_args.extend(
                [
                    "-loglevel",
                    "fatal",
                    "-nostdin",
                    "-i",
                    fname,
                    "-map",
                    "{}".format(stream),
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
                error_msg = "Stream {} not found".format(self.ref_stream)
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
                            "ffprobe", ffmpeg_resources_path=self.ffmpeg_path
                        ),
                    )["format"]["duration"]
                )
                - self.start_seconds
            )
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if "webrtc" in self.vad:
            detector = _make_webrtcvad_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "auditok" in self.vad:
            detector = _make_auditok_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "silero" in self.vad:
            detector = _make_silero_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        else:
            raise ValueError("unknown vad: %s" % self.vad)
        media_bstring: List[np.ndarray] = []
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", ffmpeg_resources_path=self.ffmpeg_path
            )
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

        with tqdm.tqdm(total=total_duration) as pbar:
            while True:
                in_bytes = process.stdout.read(
                    frames_per_window * windows_per_buffer
                )
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
                if "silero" not in self.vad:
                    in_bytes = np.frombuffer(in_bytes, np.uint8)
                media_bstring.append(detector(in_bytes))
        process.wait()
        if len(media_bstring) == 0:
            raise ValueError(
                "Unable to detect speech. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
        self.video_speech_results_ = np.concatenate(media_bstring)
        logger.info("total of speech segments: %s", np.sum(self.video_speech_results_))
        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


_PAIRED_NESTER: Dict[str, str] = {
    "(": ")",
    "{": "}",
    "[": "]",
    # FIXME: False positive sometimes when there are html tags, e.g. <i> Hello? </i>
    # '<': '>',
}


# TODO: need way better metadata detector
def _is_metadata(content: str, is_beginning_or_end: bool) -> bool:
    content = content.strip()
    if len(content) == 0:
        return True
    if (
        content[0] in _PAIRED_NESTER.keys()
        and content[-1] == _PAIRED_NESTER[content[0]]
    ):
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
        super(SubtitleSpeechTransformer, self).__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.framerate_ratio: float = framerate_ratio
        self.subtitle_speech_results_: Optional[np.ndarray] = None
        self.max_time_: Optional[float] = None # Store as float seconds

    def fit(self, subs: List[GenericSubtitle], *_) -> "SubtitleSpeechTransformer":
        max_reasonable_time_sec = 10 * 60 * 60 # 10 hours limit
        max_time_sec = 0.0
        subs_processed = 0
        subs_skipped = 0
        for sub in subs:
            sub_end_sec = sub.end.total_seconds()
            # Add check for unreasonable end times
            if sub_end_sec > max_reasonable_time_sec:
                logger.warning(f"Subtitle end time {sub.end} exceeds limit ({max_reasonable_time_sec}s), skipping for max_time calculation.")
                subs_skipped += 1
                continue # Skip this sub for max_time calculation
            max_time_sec = max(max_time_sec, sub_end_sec)
            subs_processed += 1

        if subs_processed == 0 and subs_skipped > 0:
             logger.error("All subtitles had timestamps exceeding the reasonable limit. Cannot process.")
             # Set results to empty/default to avoid downstream errors?
             self.subtitle_speech_results_ = np.array([], dtype=float)
             self.max_time_ = 0.0
             self.start_frame_ = 0
             self.end_frame_ = 0
             return self
        elif subs_processed == 0:
             logger.warning("No valid subtitles found to process.")
             self.subtitle_speech_results_ = np.array([], dtype=float)
             self.max_time_ = 0.0
             self.start_frame_ = 0
             self.end_frame_ = 0
             return self


        self.max_time_ = max_time_sec - self.start_seconds
        # Ensure max_time_ is not negative if start_seconds is large
        self.max_time_ = max(0.0, self.max_time_)

        # Calculate array size based on the determined max_time_sec
        array_size = int(round(self.max_time_ * self.sample_rate)) + 2
        try:
            samples = np.zeros(array_size, dtype=float)
        except MemoryError:
             logger.error(f"MemoryError: Failed to allocate array of size {array_size} (max_time={self.max_time_:.2f}s). Check subtitle timestamps.")
             raise # Re-raise the memory error
        except ValueError as e: # Catches negative sizes if max_time_ somehow becomes negative after rounding
             logger.error(f"ValueError creating array: size={array_size}, max_time={self.max_time_:.2f}s. Error: {e}")
             raise

        start_frame = float("inf")
        end_frame = 0
        for i, sub in enumerate(subs):
            if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)):
                continue
            # Skip metadata and previously skipped subs
            if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)) or sub.end.total_seconds() > max_reasonable_time_sec:
                continue

            start_sec = sub.start.total_seconds() - self.start_seconds
            end_sec = sub.end.total_seconds() - self.start_seconds

            # Ensure times are not negative after subtracting start_seconds
            start_sec = max(0.0, start_sec)
            end_sec = max(0.0, end_sec)

            # Ensure end is not before start
            if end_sec < start_sec:
                 logger.warning(f"Subtitle has end time before start time after offset: Start={start_sec:.3f}s, End={end_sec:.3f}s. Skipping.")
                 continue

            start_sample = int(round(start_sec * self.sample_rate))
            end_sample = int(round(end_sec * self.sample_rate))

            # Clamp indices to be within the allocated array bounds
            start_sample = max(0, min(start_sample, array_size - 1))
            end_sample = max(0, min(end_sample, array_size - 1)) # end is exclusive for slicing

            if start_sample >= end_sample: # Skip zero or negative duration samples
                 continue

            start_frame = min(start_frame, start_sample)
            end_frame = max(end_frame, end_sample) # end_frame tracks the max *exclusive* index used

            # Fill the samples array, ensuring indices are valid
            samples[start_sample:end_sample] = min(1.0 / self.framerate_ratio, 1.0)

        self.subtitle_speech_results_ = samples
        # Handle case where no valid frames were found
        if start_frame == float("inf"):
             self.start_frame_ = 0
             self.end_frame_ = 0
        else:
             self.fit_boundaries(self.subtitle_speech_results_)
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.subtitle_speech_results_ is not None
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self, non_speech_label: float) -> None:
        super(DeserializeSpeechTransformer, self).__init__()
        self._non_speech_label: float = non_speech_label
        self.deserialized_speech_results_: Optional[np.ndarray] = None

    def fit(self, fname, *_) -> "DeserializeSpeechTransformer":
        speech = np.load(fname)
        if hasattr(speech, "files"):
            if "speech" in speech.files:
                speech = speech["speech"]
            else:
                raise ValueError(
                    'could not find "speech" array in '
                    "serialized file; only contains: %s" % speech.files
                )
        speech[speech < 1.0] = self._non_speech_label
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.deserialized_speech_results_ is not None
        return self.deserialized_speech_results_
