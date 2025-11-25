import io
import logging
import subprocess
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Union, cast

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
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleScaler

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _get_silero_model_path() -> Path:
    """
    Get or download the Silero VAD ONNX model.

    Returns the path to the cached model file, downloading if necessary.
    """
    # Determine cache directory
    cache_dir = Path.home() / ".cache" / "ffsubsync"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "silero_vad.onnx"

    # Return if model already exists
    if model_path.exists():
        logger.debug(f"Using cached Silero VAD model: {model_path}")
        return model_path

    # Download model
    model_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    logger.info(f"Downloading Silero VAD ONNX model from {model_url}...")

    try:
        import requests

        response = requests.get(model_url, timeout=30)
        response.raise_for_status()

        # Write to temp file first, then rename (atomic operation)
        temp_path = model_path.with_suffix(".onnx.tmp")
        temp_path.write_bytes(response.content)
        temp_path.rename(model_path)

        logger.info(f"Silero VAD model downloaded successfully: {model_path}")
        return model_path

    except ImportError:
        raise ImportError(
            "requests library is required to download Silero VAD model. "
            "Install with: pip install ffsubsync[silero]"
        ) from None
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Silero VAD model from {model_url}: {e}"
        ) from e


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
            ADSFactory,
            AudioEnergyValidator,
            BufferAudioSource,
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


def _smooth_vad_output(signal: np.ndarray, iterations: int) -> np.ndarray:
    """
    Apply binary dilation to fill small gaps in VAD output.

    This makes the audio fingerprint look more like subtitle fingerprint
    by filling gaps shorter than iterations * 10ms (at 100 sample_rate).

    Uses numpy convolution to avoid scipy dependency.
    """
    if iterations <= 0:
        return signal

    # Convert to binary
    binary = signal > 0.5

    # Apply dilation using convolution with a kernel of ones
    # Kernel size: 2 * iterations + 1 (symmetric around center)
    kernel_size = 2 * iterations + 1
    kernel = np.ones(kernel_size, dtype=float)

    # Convolve: any non-zero value means there was speech within the window
    result = np.convolve(binary.astype(float), kernel, mode="same")

    # Convert back to binary (any overlap with the kernel means speech)
    return (result > 0).astype(float)


def _make_webrtcvad_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
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
            # webrtcvad has low recall on mode 3, so treat non-speech as "not sure"
            media_bstring.append(1.0 if is_speech else non_speech_label)

        result = np.array(media_bstring)

        # Fill gaps shorter than 300ms (30 frames at 100 sample_rate)
        # This matches subtitle characteristics better than raw VAD
        smoothed = _smooth_vad_output(result, iterations=30)

        return smoothed

    return _detect


def _make_silero_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    """
    Create Silero VAD detector using ONNX Runtime.

    Silero VAD is a stateful model that requires hidden states (h, c) to be
    passed between inference calls. Uses 512-sample chunks at 16kHz.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for Silero VAD. "
            "Install with: pip install ffsubsync[silero]"
        ) from None

    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)

    # Get or download the ONNX model
    model_path = _get_silero_model_path()

    # Initialize ONNX Runtime session
    # Use CPUExecutionProvider for best compatibility on ARM64
    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=providers
    )

    logger.info("Silero VAD ONNX backend initialized (CPUExecutionProvider)")

    # Silero VAD requires EXACTLY 512 samples per chunk at 16kHz (32ms)
    silero_chunk_size = 512
    # Silero also requires 64-sample context window at 16kHz
    context_size = 64

    # Initialize hidden state for the stateful model
    # state has shape (2, 1, 128) for batch size 1
    state = np.zeros((2, 1, 128), dtype=np.float32)
    # Initialize context buffer
    context = np.zeros((1, context_size), dtype=np.float32)

    exception_logged = False

    def _detect(asegment: bytes) -> np.ndarray:
        nonlocal state, context, exception_logged

        # Convert bytes to float32 audio
        audio = np.frombuffer(asegment, np.int16).astype(np.float32) / 32768.0

        # Calculate number of our output windows
        num_windows = (len(audio) + frames_per_window - 1) // frames_per_window

        # Process audio in 512-sample chunks for Silero
        silero_probs = []
        for i in range(0, len(audio), silero_chunk_size):
            chunk = audio[i : i + silero_chunk_size]

            # Pad last chunk if necessary
            if len(chunk) < silero_chunk_size:
                padded = np.zeros(silero_chunk_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded

            try:
                # Concatenate context + chunk as required by Silero ONNX
                # Input shape must be (1, context_size + chunk_size) = (1, 64 + 512) = (1, 576)
                input_with_context = np.concatenate(
                    [context, chunk.reshape(1, silero_chunk_size)], axis=1
                ).astype(np.float32)

                # Prepare ONNX inputs
                # input: (1, 576) - context + audio chunk
                # sr: int64 scalar - sample rate
                # state: (2, 1, 128) - hidden state
                ort_inputs = {
                    "input": input_with_context,
                    "sr": np.array(frame_rate, dtype=np.int64),
                    "state": state,
                }

                # Run inference
                # Returns: [output, stateN]
                outputs = session.run(None, ort_inputs)
                speech_prob = float(outputs[0][0][0])  # Extract scalar probability

                # Update hidden state for next iteration (stateful!)
                state = outputs[1]

                # Update context to be the last context_size samples
                context = input_with_context[:, -context_size:]

            except Exception:
                if not exception_logged:
                    exception_logged = True
                    logger.exception("exception occurred during speech detection")
                speech_prob = 0.0

            silero_probs.append(speech_prob)

        # Map Silero chunk probabilities to our smaller windows
        # Each Silero chunk covers multiple windows
        windows_per_silero_chunk = silero_chunk_size / frames_per_window
        media_bstring = []

        for window_idx in range(num_windows):
            # Find which Silero chunk this window belongs to
            silero_idx = int(window_idx / windows_per_silero_chunk)
            silero_idx = min(silero_idx, len(silero_probs) - 1)  # Clamp to valid range

            speech_prob = silero_probs[silero_idx]

            # Blend probability with non_speech_label
            blended_prob = 1.0 - (1.0 - speech_prob) * (1.0 - non_speech_label)
            media_bstring.append(blended_prob)

        return np.array(media_bstring)

    return _detect


def _make_tenvad_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    """
    Create a detector using TEN VAD.

    Notes
    - TEN VAD expects 16 kHz audio and a hop size of either 160 (10 ms)
      or 256 (16 ms). We derive hop size from the requested window length
      implied by `sample_rate` and `frame_rate`.
    - Returns per-window speech probabilities in [0, 1].
    """
    try:
        from ten_vad import TenVad  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        # Fallback to ONNX-based implementation (ARM64 compatible)
        try:
            from ffsubsync.ten_vad_onnx import TenVadONNX as TenVad  # type: ignore

            logger.info(
                "Using ONNX-based TEN-VAD backend (ARM64/aarch64 compatible). "
                "For better performance on x64 platforms, install: pip install ten-vad"
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error: Neither ten-vad nor ONNX backend available.\n"
                "Install with: pip install ffsubsync[tenvad-onnx]\n"
                "Or for x64 platforms: pip install ten-vad\n"
                "TEN VAD requires 16 kHz audio."
            )
            raise e

    # Window duration in seconds and derived hop size in samples
    window_duration = 1.0 / sample_rate
    frames_per_window = int(window_duration * frame_rate + 0.5)

    # Instantiate detector with derived hop size and a default threshold.
    # Threshold only affects the binary flag; we will consume the probability.
    threshold = 0.5
    ten = TenVad(frames_per_window, threshold)

    def _detect(asegment: bytes) -> np.ndarray:
        # View as int16 without copying
        pcm = np.frombuffer(asegment, dtype=np.int16)
        n = len(pcm)
        media_bstring: list[float] = []
        # Process in hop-sized chunks
        for start in range(0, n, frames_per_window):
            stop = min(start + frames_per_window, n)
            # If final chunk shorter than hop, pad with zeros to expected length
            chunk = pcm[start:stop]
            if len(chunk) < frames_per_window:
                padded = np.zeros(frames_per_window, dtype=np.int16)
                padded[: len(chunk)] = chunk
                chunk = padded
            try:
                prob, _ = ten.process(chunk)  # returns (probability, 0/1)
            except Exception:  # pragma: no cover
                # Be conservative on exception; treat as non-speech but not certain
                prob = 0.0
            # Blend with non_speech_label similar to Silero path
            media_bstring.append(1.0 - (1.0 - float(prob)) * (1.0 - non_speech_label))
        return np.array(media_bstring, dtype=float)

    return _detect


class WhisperSpeechTransformer(TransformerMixin):
    def __init__(
        self,
        sample_rate: int,
        frame_rate: int,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        vlc_mode: bool = False,
        max_transcription_seconds: float = 300.0,
    ) -> None:
        super().__init__()
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.vlc_mode: bool = vlc_mode
        self.max_transcription_seconds: float = max_transcription_seconds
        self.video_speech_results_: Optional[np.ndarray] = None

    def fit(self, fname: str, *_) -> "WhisperSpeechTransformer":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for Whisper VAD. "
                "Install with: pip install ffsubsync[whisper] or pip install faster-whisper"
            ) from None

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
            logger.info("Extracting audio for Whisper VAD...")

            ffmpeg_args = [
                ffmpeg_bin_path("ffmpeg", ffmpeg_resources_path=self.ffmpeg_path)
            ]
            if self.start_seconds > 0:
                ffmpeg_args.extend(["-ss", str(timedelta(seconds=self.start_seconds))])

            ffmpeg_args.extend(
                [
                    "-y",
                    "-loglevel",
                    "fatal",
                    "-nostdin",
                    "-i",
                    fname,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                ]
            )

            # We can limit the duration of extraction too, to save time!
            # But user didn't explicitly ask for that, and we might want to keep it simple.
            # Actually, if we only need 5 mins, we should only extract 5 mins.
            # Let's add -t to ffmpeg if max_transcription_seconds is set.
            if self.max_transcription_seconds:
                ffmpeg_args.extend(["-t", str(self.max_transcription_seconds)])

            ffmpeg_args.extend(["-f", "wav", tmp_audio.name])
            subprocess.check_call(ffmpeg_args, **subprocess_args(include_stdout=False))

            logger.info("Loading faster-whisper model 'tiny'...")
            # device="cpu" and compute_type="int8" are good defaults for speed on most machines including ARM64
            model = WhisperModel("tiny", device="cpu", compute_type="int8")

            logger.info("Transcribing audio...")
            # vad_filter=True helps remove non-speech noises even within segments
            # But it might be too aggressive for speech over music. Let's try False.
            segments, info = model.transcribe(tmp_audio.name, vad_filter=False)

            # info.duration is the duration of the audio file we passed in
            # We must use self.sample_rate (100Hz) for the output array, NOT self.frame_rate (audio rate)
            duration_to_use = info.duration
            if (
                self.max_transcription_seconds
                and duration_to_use > self.max_transcription_seconds
            ):
                duration_to_use = self.max_transcription_seconds

            total_frames = int(duration_to_use * self.sample_rate)
            media_bstring = np.zeros(
                total_frames + 100, dtype=float
            )  # Add a bit of padding just in case

            count = 0
            for segment in segments:
                if (
                    self.max_transcription_seconds
                    and segment.end > self.max_transcription_seconds
                ):
                    break

                # Filter out music/sound markers if possible
                text = segment.text.strip()
                if text.startswith("[") and text.endswith("]"):
                    logger.info(
                        f"Ignored non-speech segment: {text} at {segment.start}-{segment.end}"
                    )
                    continue

                logger.info(f"Speech segment: {text} at {segment.start}-{segment.end}")

                start_frame = int(segment.start * self.sample_rate)
                end_frame = int(segment.end * self.sample_rate)
                # Clip to array bounds
                start_frame = max(0, start_frame)
                end_frame = min(len(media_bstring), end_frame)

                if end_frame > start_frame:
                    media_bstring[start_frame:end_frame] = 1.0
                count += 1

            # Trim to actual duration if needed, but having a bit extra is usually fine.
            # The aligner might complain if sizes are vastly different, but usually it handles it.
            # Let's trim to the exact frame count expected from info.duration
            media_bstring = media_bstring[:total_frames]

            logger.info(f"Detected {count} speech segments.")
            self.video_speech_results_ = media_bstring

        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


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
    ) -> None:
        super().__init__()
        self.vad: str = vad
        self.sample_rate: int = sample_rate
        # TEN VAD requires 16 kHz input. If selected, force 16kHz for extraction.
        if "tenvad" in vad and frame_rate != 16000:
            logger.info(
                "TEN VAD selected: overriding frame_rate %d -> 16000 for extraction.",
                frame_rate,
            )
            self.frame_rate = 16000
        # Silero VAD also requires 16 kHz (or 8 kHz) input
        elif "silero" in vad and frame_rate != 16000:
            logger.info(
                "Silero VAD selected: overriding frame_rate %d -> 16000 for extraction.",
                frame_rate,
            )
            self.frame_rate = 16000
        # WebRTC VAD: speech information is fully contained in 16kHz, so downsample for 3x speedup
        elif "webrtc" in vad and frame_rate > 16000:
            logger.info(
                "WebRTC VAD selected: reducing frame_rate from %d to 16000 for 3x speedup.",
                frame_rate,
            )
            self.frame_rate = 16000
        else:
            self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.video_speech_results_: Optional[np.ndarray] = None

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

    def _build_detector(self) -> Callable[[bytes], np.ndarray]:
        if "webrtc" in self.vad:
            return _make_webrtcvad_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        if "auditok" in self.vad:
            return _make_auditok_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        if "silero" in self.vad:
            return _make_silero_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        if "tenvad" in self.vad:
            try:
                return _make_tenvad_detector(
                    self.sample_rate, self.frame_rate, self._non_speech_label
                )
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "TEN VAD unavailable (%s); falling back to WebRTC VAD.", exc
                )
                return _make_webrtcvad_detector(
                    self.sample_rate, self.frame_rate, self._non_speech_label
                )
        raise ValueError(f"unknown vad: {self.vad}")

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
        detector = self._build_detector()
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


_PAIRED_NESTER: dict[str, str] = {
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
    if content[0] in _PAIRED_NESTER and content[-1] == _PAIRED_NESTER[content[0]]:
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
        self.subtitle_speech_results_: Optional[np.ndarray] = None
        self.max_time_: Optional[int] = None

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
    def __init__(self, non_speech_label: float) -> None:
        super().__init__()
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
                    f"serialized file; only contains: {speech.files}"
                )
        speech[speech < 1.0] = self._non_speech_label
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.deserialized_speech_results_ is not None
        return self.deserialized_speech_results_
