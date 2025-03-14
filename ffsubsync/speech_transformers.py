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
import pickle
import tempfile
import multiprocessing

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
        vlc_mode: bool = False,
        gui_mode: bool = False,
    ) -> None:
        super(VideoSpeechTransformer, self).__init__()
        self.vad: str = vad
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.gui_mode: bool = gui_mode
        self.video_speech_results_: Optional[np.ndarray] = None

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        import multiprocessing
        cpu_count = min(2, multiprocessing.cpu_count())
        
        # Create cache directory
        cache_dir = os.path.join(os.path.dirname(fname), ".ffsubsync_cache")
        os.makedirs(cache_dir, exist_ok=True)
        subs_cache_file = os.path.join(cache_dir, f"{os.path.basename(fname)}.subs_cache")
        
        # Check for cached results
        if os.path.exists(subs_cache_file):
            try:
                with open(subs_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.video_speech_results_ = cached_data
                    logger.info("Using cached subtitle extraction results")
                    return
            except Exception as e:
                logger.warning(f"Failed to load subtitle cache: {e}")
        
        embedded_subs = []
        embedded_subs_times = []
        
        # Optimize FFmpeg command for subtitle extraction
        ffmpeg_base_args = [
            ffmpeg_bin_path("ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path),
            "-loglevel", "fatal", 
            "-nostdin", 
            "-threads", str(cpu_count),  # Limit FFmpeg threads
            "-y", 
            "-i", fname
        ]
        
        # Use a more efficient approach for subtitle extraction
        # Instead of extracting all subtitles at once, extract them one by one
        video_dir = os.path.dirname(fname)
        subs_dir = os.path.join(video_dir, "Subs")
        os.makedirs(subs_dir, exist_ok=True)
        
        # First, get information about subtitle streams
        ffprobe_args = [
            ffmpeg_bin_path("ffprobe", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path),
            "-v", "error", 
            "-select_streams", "s", 
            "-show_entries", "stream=index:stream_tags=language,title", 
            "-of", "csv=p=0", 
            fname
        ]
        
        logger.debug(f"Running ffprobe command: {' '.join(ffprobe_args)}")
        process = subprocess.Popen(ffprobe_args, **subprocess_args(include_stdout=True))
        output = process.communicate()[0].decode("utf-8").strip().splitlines()
        
        if process.returncode != 0 or not output:
            raise ValueError("Failed to detect subtitle streams in the video file")
        
        # Process each subtitle stream separately
        for line in output:
            parts = line.split(",")
            if len(parts) < 2:
                logger.warning(f"Unexpected ffprobe output: {line}")
                continue
            
            stream_index, language_code = parts[:2]
            language_title = parts[2] if len(parts) > 2 else None
            
            subtitle_filename = os.path.join(
                subs_dir, 
                f"{os.path.splitext(os.path.basename(fname))[0]}.{language_code}"
                + (f".{language_title}" if language_title else "")
                + ".srt"
            )
            
            # Check if this subtitle file already exists
            if os.path.exists(subtitle_filename):
                logger.info(f"Using existing subtitle file: {subtitle_filename}")
            else:
                # Extract just this subtitle stream
                stream_ffmpeg_args = [
                    ffmpeg_bin_path("ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path),
                    "-loglevel", "fatal", 
                    "-nostdin", 
                    "-threads", str(cpu_count),
                    "-y", 
                    "-i", fname,
                    "-map", f"0:{stream_index}", 
                    "-f", "srt", 
                    subtitle_filename
                ]
                
                logger.debug(f"Running ffmpeg command: {' '.join(stream_ffmpeg_args)}")
                process = subprocess.Popen(stream_ffmpeg_args, **subprocess_args(include_stdout=True))
                process.communicate()
                
                if process.returncode != 0:
                    logger.warning(f"Failed to extract subtitle stream {stream_index}")
                    continue
            
            # Process this subtitle file
            try:
                with open(subtitle_filename, 'rb') as file:
                    pipe = cast(Pipeline, make_subtitle_speech_pipeline(start_seconds=self.start_seconds))
                    pipe_output = pipe.fit(file)
                    for speech_step in pipe_output.steps[-1:]:
                        embedded_subs.append(speech_step[1])
                        embedded_subs_times.append(speech_step[1].max_time_)
            except Exception as e:
                logger.warning(f"Failed to process subtitle file {subtitle_filename}: {e}")
        
        if not embedded_subs:
            error_msg = "Video file appears to lack subtitle stream" if self.ref_stream is None else f"Stream {self.ref_stream} not found"
            raise ValueError(error_msg)
        
        subs_to_use = embedded_subs[int(np.argmax(embedded_subs_times))]
        self.video_speech_results_ = subs_to_use.subtitle_speech_results_
        
        # Cache the results
        try:
            with open(subs_cache_file, 'wb') as f:
                pickle.dump(self.video_speech_results_, f)
        except Exception as e:
            logger.warning(f"Failed to cache subtitle results: {e}")

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
                            self.gui_mode,
                            ffmpeg_resources_path=self.ffmpeg_path,
                        ),
                    )["format"]["duration"]
                )
                - self.start_seconds
            )
            logger.info(f"Video total duration: {total_duration:.2f} seconds")
            
            # Log processing range
            if self.start_seconds > 0:
                logger.info(f"Starting processing at {self.start_seconds:.2f} seconds")
                
            # Log VAD method being used
            logger.info(f"Using VAD method: {self.vad}")
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
        cpu_count = min(2, multiprocessing.cpu_count())  # Limit to available cores
        media_bstring: List[np.ndarray] = []
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            )
        ]
        if self.start_seconds > 0:
            ffmpeg_args.extend(
                [
                    "-ss",
                    str(timedelta(seconds=self.start_seconds)),
                ]
            )
        ffmpeg_args.extend([
            "-loglevel", "fatal", 
            "-nostdin", 
            "-threads", str(cpu_count),  # Limit FFmpeg threads
            "-i", fname
        ])
        if self.ref_stream is not None and self.ref_stream.startswith("0:a:"):
            ffmpeg_args.extend(["-map", self.ref_stream])
        ffmpeg_args.extend([
            # Optimize audio processing for speech detection
            "-af", "aresample=8000",  # Downsample to 8kHz for speech detection
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(self.frame_rate),
            "-"
        ])
        
        # Log that we're starting ffmpeg
        logger.info(f"Starting ffmpeg processing for speech extraction...")
        
        # Add a timeout mechanism for ffmpeg to prevent hanging
        import time
        import signal
        
        start_time = time.time()
        max_process_time = 600  # 10 minutes max for a segment
        
        # Create process with appropriate timeouts
        process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
        
        bytes_per_frame = 2
        frames_per_window = bytes_per_frame * self.frame_rate // self.sample_rate
        windows_per_buffer = 1000
        simple_progress = 0.0

        # Import ProgressReporter to get the appropriate progress display
        from ffsubsync.ffsubsync import ProgressReporter
        
        # Check if we're in a worker process
        is_worker = ProgressReporter.is_worker_process()
        
        # For worker processes, use simpler progress reporting approach
        if is_worker:
            try:
                # Processing loop with timeouts and periodic updates
                last_update_time = time.time()
                update_interval = 2.0  # Update every 2 seconds
                
                while True:
                    # Set a timeout for reading from ffmpeg
                    if process.poll() is not None:
                        # Process ended
                        break
                    
                    # Check overall timeout
                    if time.time() - start_time > max_process_time:
                        logger.warning("FFmpeg processing timeout exceeded, terminating")
                        process.terminate()
                        time.sleep(0.5)
                        if process.poll() is None:
                            process.kill()
                        raise TimeoutError("FFmpeg processing took too long")
                    
                    try:
                        # Try to read with a timeout
                        in_bytes = process.stdout.read(frames_per_window * windows_per_buffer)
                        if not in_bytes:
                            break
                        
                        newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                        simple_progress += newstuff
                        
                        # Provide periodic progress updates through logger
                        current_time = time.time()
                        if current_time - last_update_time > update_interval:
                            if total_duration is not None:
                                progress_pct = min(100, simple_progress * 100.0 / total_duration)
                                logger.debug(f"Extraction progress: {progress_pct:.1f}% ({simple_progress:.1f}/{total_duration:.1f}s)")
                            else:
                                logger.debug(f"Extraction progress: {simple_progress:.1f}s")
                            last_update_time = current_time
                        
                        if "silero" not in self.vad:
                            in_bytes = np.frombuffer(in_bytes, np.uint8)
                            media_bstring.append(detector(in_bytes))
                        else:
                            # Silero processing needs to be in smaller chunks
                            for i in range(0, len(in_bytes), frames_per_window):
                                chunk = in_bytes[i:i + frames_per_window]
                                if len(chunk) == frames_per_window:  # Only process complete chunks
                                    media_bstring.append(detector(chunk))
                    except Exception as e:
                        logger.error(f"Error during extraction: {e}")
                        break
            except Exception as e:
                logger.error(f"Error in extraction loop: {e}")
                raise
            finally:
                # Clean up process if still running
                if process.poll() is None:
                    process.terminate()
                    time.sleep(0.1)
                    if process.poll() is None:
                        process.kill()
        else:
            # For main process, create a proper progress display
            try:
                # Create a dedicated progress display
                Progress = ProgressReporter.get_progress_class()
                with Progress(
                    "[progress.description]{task.description}",
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "{task.completed:.1f}/{task.total:.1f}s",
                    transient=False
                ) as progress:
                    # Add task for speech extraction
                    task_id = progress.add_task(
                        "Extracting speech", 
                        total=total_duration if total_duration is not None else 100.0,
                        completed=0.0
                    )
                    
                    # Processing loop with progress updates
                    last_update_time = time.time()
                    update_interval = 0.5  # Update every 0.5 seconds for smoother progress
                    
                    while True:
                        # Check if process ended
                        if process.poll() is not None:
                            break
                        
                        # Check timeout
                        if time.time() - start_time > max_process_time:
                            logger.warning("FFmpeg processing timeout exceeded, terminating")
                            process.terminate()
                            time.sleep(0.5)
                            if process.poll() is None:
                                process.kill()
                            raise TimeoutError("FFmpeg processing took too long")
                        
                        try:
                            # Read from ffmpeg
                            in_bytes = process.stdout.read(frames_per_window * windows_per_buffer)
                            if not in_bytes:
                                break
                            
                            # Calculate progress
                            newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                            simple_progress += newstuff
                            
                            # Update progress bar regularly
                            current_time = time.time()
                            if current_time - last_update_time > update_interval:
                                progress.update(task_id, completed=simple_progress)
                                last_update_time = current_time
                            
                            # Process audio data
                            if "silero" not in self.vad:
                                in_bytes = np.frombuffer(in_bytes, np.uint8)
                                media_bstring.append(detector(in_bytes))
                            else:
                                # Silero processing needs to be in smaller chunks
                                for i in range(0, len(in_bytes), frames_per_window):
                                    chunk = in_bytes[i:i + frames_per_window]
                                    if len(chunk) == frames_per_window:  # Only process complete chunks
                                        media_bstring.append(detector(chunk))
                        except Exception as e:
                            logger.error(f"Error during extraction: {e}")
                            break
                            
                    # Final progress update
                    progress.update(task_id, completed=simple_progress)
            except Exception as e:
                logger.error(f"Error in progress display: {e}")
                raise
            finally:
                # Clean up process if still running
                if process.poll() is None:
                    process.terminate()
                    time.sleep(0.1)
                    if process.poll() is None:
                        process.kill()
            
        # Don't wait for process if already terminated
        if process.poll() is None:
            process.wait()
            
        logger.info(f"FFmpeg processing complete. Processed {simple_progress:.2f} seconds.")
            
        if len(media_bstring) == 0:
            raise ValueError(
                "Unable to detect speech. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
        self.video_speech_results_ = np.concatenate(media_bstring)
        logger.info("Total of speech segments: %s", np.sum(self.video_speech_results_))
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
        self.max_time_: Optional[int] = None

    def fit(self, subs: List[GenericSubtitle], *_) -> "SubtitleSpeechTransformer":
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
            start = int(
                round(
                    (sub.start.total_seconds() - self.start_seconds) * self.sample_rate
                )
            )
            start_frame = min(start_frame, start)
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            end_frame = max(end_frame, end)
            samples[start:end] = min(1.0 / self.framerate_ratio, 1.0)
        self.subtitle_speech_results_ = samples
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
