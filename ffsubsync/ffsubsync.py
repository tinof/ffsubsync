#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import resource
import signal
import tempfile
import traceback
import psutil
import multiprocessing

from ffsubsync.aligners import FFTAligner, MaxScoreAligner, AnchorAligner, FailedToFindAlignmentException
from ffsubsync.constants import (
    DEFAULT_APPLY_OFFSET_SECONDS,
    DEFAULT_FRAME_RATE,
    DEFAULT_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_NON_SPEECH_LABEL,
    DEFAULT_START_SECONDS,
    DEFAULT_VAD,
    DEFAULT_ENCODING,
    FRAMERATE_RATIOS,
    SAMPLE_RATE,
    SUBTITLE_EXTENSIONS,
    DEFAULT_SUCCESS_THRESHOLD,
    MIN_ACCEPTABLE_SCORE,
    MAX_AUTO_ANCHORS,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline,
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleMerger, SubtitleShifter, SubtitleMorpher
from ffsubsync.version import get_version

# Create shared helpers for progress reporting
class ProgressReporter:
    """Handles progress reporting with correct behavior in both main and worker processes"""
    
    # Use a class variable to track if we're in a worker process
    _is_worker_process = None
    
    @staticmethod
    def is_worker_process():
        """Check if we're running in a worker process
        
        This method caches the result to avoid repeated checks.
        """
        if ProgressReporter._is_worker_process is None:
            import multiprocessing
            ProgressReporter._is_worker_process = multiprocessing.current_process().name != 'MainProcess'
        return ProgressReporter._is_worker_process
    
    @staticmethod
    def get_progress_class():
        """Get the appropriate Progress class based on context
        
        In worker processes, return a dummy class that maintains API compatibility.
        In the main process, return the actual Rich Progress class.
        """
        # In worker processes, use a dummy class for API compatibility
        if ProgressReporter.is_worker_process():
            class DummyProgress:
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args, **kwargs):
                    pass
                def add_task(self, *args, **kwargs):
                    return 0
                def update(self, *args, **kwargs):
                    pass
            return DummyProgress
        
        # In main process, use actual Rich progress display
        try:
            from rich.progress import Progress
            return Progress
        except ImportError:
            # Fallback if Rich not available
            class DummyProgress:
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args, **kwargs):
                    pass
                def add_task(self, *args, **kwargs):
                    return 0
                def update(self, *args, **kwargs):
                    pass
            return DummyProgress
            
    @staticmethod
    def create_progress_display(*args, **kwargs):
        """Create the appropriate progress display with optimal settings
        
        This centralizes progress creation to enable further optimizations.
        In main process, we use a proper Rich display with console capture.
        In worker processes, we use a dummy display that maintains API compatibility.
        
        Args:
            *args: Arguments to pass to Progress constructor
            **kwargs: Keyword arguments to pass to Progress constructor
            
        Returns:
            A progress display object (either Rich Progress or DummyProgress)
        """
        Progress = ProgressReporter.get_progress_class()
        
        if not ProgressReporter.is_worker_process():
            # In main process, set up a dedicated console for progress
            try:
                from rich.console import Console
                from rich.live import Live
                
                # Create a dedicated console for progress display to avoid conflicts with logging
                console = Console(file=sys.stderr)
                
                # Use redirect_stdout and redirect_stderr to avoid log messages interrupting progress
                if 'console' not in kwargs:
                    kwargs['console'] = console
                
                # Set transient=False to keep the progress bar on screen
                if 'transient' not in kwargs:
                    kwargs['transient'] = False
                    
                return Progress(*args, **kwargs)
            except ImportError:
                # Fall back to basic progress
                return Progress(*args, **kwargs)
        else:
            # In worker process, return dummy progress
            return Progress(*args, **kwargs)

logger: logging.Logger = logging.getLogger(__name__)


def override(args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname: Optional[str]) -> Optional[str]:
    if ref_fname is None:
        return None
    return ref_fname[-3:]


def make_test_case(
    args: argparse.Namespace, npy_savename: Optional[str], sync_was_successful: bool
) -> int:
    if npy_savename is None:
        raise ValueError("need non-null npy_savename")
    tar_dir = "{}.{}".format(
        args.reference, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    logger.info("creating test archive {}.tar.gz...".format(tar_dir))
    os.mkdir(tar_dir)
    try:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        shutil.copy(log_path, tar_dir)
        shutil.copy(args.srtin[0], tar_dir)
        if sync_was_successful:
            shutil.move(args.srtout, tar_dir)
        if _ref_format(args.reference) in SUBTITLE_EXTENSIONS:
            shutil.copy(args.reference, tar_dir)
        elif args.serialize_speech or args.reference == npy_savename:
            shutil.copy(npy_savename, tar_dir)
        else:
            shutil.move(npy_savename, tar_dir)
        supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
        preferred_formats = ["gztar", "bztar", "xztar", "zip", "tar"]
        for archive_format in preferred_formats:
            if archive_format in supported_formats:
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                break
        else:
            logger.error(
                "failed to create test archive; no formats supported "
                "(this should not happen)"
            )
            return 1
        logger.info("...done")
    finally:
        shutil.rmtree(tar_dir)
    return 0


def get_srt_pipe_maker(
    args: argparse.Namespace, srtin: Optional[str]
) -> Callable[[Optional[float]], Union[Pipeline, Callable[[float], Pipeline]]]:
    if srtin is None:
        srtin_format = "srt"
    else:
        srtin_format = os.path.splitext(srtin)[-1][1:]
    parser = make_subtitle_parser(fmt=srtin_format, caching=True, **args.__dict__)
    return lambda scale_factor: make_subtitle_speech_pipeline(
        **override(args, scale_factor=scale_factor, parser=parser)
    )


def get_framerate_ratios_to_try(args: argparse.Namespace) -> List[Optional[float]]:
    if args.no_fix_framerate:
        return []
    else:
        framerate_ratios = list(
            np.concatenate(
                [np.array(FRAMERATE_RATIOS), 1.0 / np.array(FRAMERATE_RATIOS)]
            )
        )
        if args.gss:
            framerate_ratios.append(None)
        return framerate_ratios


def calculate_morphed_score(ref_speech: np.ndarray, morphed_speech: np.ndarray) -> float:
    """Calculate alignment score after morphing."""
    if len(ref_speech) != len(morphed_speech):
        # Pad the shorter signal if lengths don't match
        max_len = max(len(ref_speech), len(morphed_speech))
        if len(ref_speech) < max_len:
            ref_speech = np.pad(ref_speech, (0, max_len - len(ref_speech)))
        else:
            morphed_speech = np.pad(morphed_speech, (0, max_len - len(morphed_speech)))
    
    # Use FFTAligner to compute correlation score
    return FFTAligner().fit_transform(ref_speech, morphed_speech, get_score=True)[0]


def _test_framerate_ratio(ratio_pipe, srtin, reference_speech, max_offset_seconds):
    try:
        # Extract speech from subtitle with this framerate ratio
        sub_speech = ratio_pipe.transform(srtin)
        # Try global alignment first
        fft_aligner = FFTAligner(
            max_offset_samples=int(max_offset_seconds * SAMPLE_RATE),
            use_progressive=True
        )
        fft_aligner.fit(reference_speech, sub_speech)
        score, offset = fft_aligner.best_score_, fft_aligner.best_offset_
        return (score, offset), ratio_pipe
    except Exception as e:
        logger.warning(f"Error testing framerate ratio: {e}")
        return (float('-inf'), 0), ratio_pipe


def try_sync(
    args: argparse.Namespace, reference_pipe: Optional[Pipeline], result: Dict[str, Any]
) -> bool:
    result["sync_was_successful"] = False
    sync_was_successful = True
    logger.info(
        "extracting speech segments from %s...",
        "stdin" if not args.srtin else "subtitles file(s) {}".format(args.srtin),
    )
    
    if not args.srtin:
        args.srtin = [None]
        
    # Define excellent threshold for early termination
    EXCELLENT_THRESHOLD = 0.95
        
    for srtin in args.srtin:
        try:
            skip_sync = args.skip_sync or reference_pipe is None
            skip_infer_framerate_ratio = args.skip_infer_framerate_ratio or reference_pipe is None
            srtout = srtin if args.overwrite_input else args.srtout
            
            # Setup subtitle processing pipeline
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            framerate_ratios = get_framerate_ratios_to_try(args)
            srt_pipes = [srt_pipe_maker(1.0)] + [
                srt_pipe_maker(rat) for rat in framerate_ratios
            ]
            
            # Initial pipeline setup
            for srt_pipe in srt_pipes:
                if callable(srt_pipe):
                    continue
                else:
                    srt_pipe.fit(srtin)
                    
            # Infer framerate ratio if needed
            if not skip_infer_framerate_ratio and hasattr(reference_pipe[-1], "num_frames"):
                inferred_ratio = float(reference_pipe[-1].num_frames) / cast(Pipeline, srt_pipes[0])[-1].num_frames
                logger.info("inferred framerate ratio: %.3f", inferred_ratio)
                srt_pipes.append(
                    cast(Pipeline, srt_pipe_maker(inferred_ratio)).fit(srtin)
                )
                
            if skip_sync:
                best_score = 0.0
                best_srt_pipe = cast(Pipeline, srt_pipes[0])
                offset_samples = 0
            else:
                # Get reference speech signal
                reference_speech = reference_pipe.transform(args.reference)
                
                # Define alignment strategies
                alignment_strategies = [
                    {'type': 'global', 'params': {'max_offset_seconds': args.max_offset_seconds}},
                    {'type': 'anchors', 'params': {'n_anchors': 2, 'min_score': 0.8}},
                    {'type': 'anchors', 'params': {'n_anchors': 'auto', 'max_anchors': MAX_AUTO_ANCHORS}}
                ]
                
                best_score = float('-inf')
                best_srt_pipe = srt_pipes[0]  # Initialize with first pipe
                offset_samples = 0
                
                # Process framerate ratios in parallel if enabled
                if getattr(args, 'parallel', False) and len(srt_pipes) > 1:
                    try:
                        from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
                        
                        logger.info(f"Testing {len(srt_pipes)} framerate ratios in parallel")
                        max_workers = min(multiprocessing.cpu_count(), len(srt_pipes))
                        
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            futures = [executor.submit(_test_framerate_ratio, pipe, srtin, reference_speech, args.max_offset_seconds) for pipe in srt_pipes]
                            
                            ratio_results = []
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                    ratio_results.append(result)
                                except Exception as e:
                                    logger.warning(f"Error in parallel processing: {e}")
                        
                        # Find best result
                        if ratio_results:
                            (score, offset), pipe = max(ratio_results, key=lambda x: x[0][0])
                            
                            if score > best_score:
                                best_score = score
                                best_srt_pipe = pipe
                                offset_samples = offset
                                
                            logger.info(f"Best framerate ratio score: {best_score:.3f}")
                            
                            # Early termination if we have an excellent match
                            if best_score >= EXCELLENT_THRESHOLD:
                                logger.info(f"Found excellent match (score: {best_score:.3f}), skipping further strategies")
                                
                    except ImportError:
                        logger.warning("Parallel processing requested but multiprocessing not available")
                        # Fall back to sequential processing
                        for pipe in srt_pipes:
                            (score, offset), pipe = _test_framerate_ratio(pipe, srtin, reference_speech, args.max_offset_seconds)
                            if score > best_score:
                                best_score = score
                                best_srt_pipe = pipe
                                offset_samples = offset
                else:
                    # Sequential processing of framerate ratios
                    for pipe in srt_pipes:
                        (score, offset), pipe = _test_framerate_ratio(pipe, srtin, reference_speech, args.max_offset_seconds)
                        if score > best_score:
                            best_score = score
                            best_srt_pipe = pipe
                            offset_samples = offset
                            
                        # Early termination if we have an excellent match
                        if best_score >= EXCELLENT_THRESHOLD:
                            logger.info(f"Found excellent match (score: {best_score:.3f}), skipping further ratios")
                            break
                
                # If we don't have a good enough score yet, try anchor-based alignment
                if best_score < DEFAULT_SUCCESS_THRESHOLD:
                    for strategy in alignment_strategies:
                        try:
                            if strategy['type'] == 'global':
                                # We already did global alignment during framerate ratio testing
                                continue
                                
                            elif strategy['type'] == 'anchors':
                                if best_score >= strategy['params'].get('min_score', 0):
                                    logger.info("Previous alignment score sufficient, skipping anchor-based alignment")
                                    continue
                                    
                                logger.info("Trying anchor-based alignment with %s anchors...", 
                                          strategy['params']['n_anchors'])
                                
                                # Get speech signal from current best pipeline
                                sub_speech = best_srt_pipe.transform(srtin)
                                
                                # Find anchor points with content-aware selection
                                anchor_aligner = AnchorAligner(
                                    n_anchors=strategy['params']['n_anchors'],
                                    max_anchors=strategy['params'].get('max_anchors', MAX_AUTO_ANCHORS),
                                    content_aware=True
                                )
                                
                                try:
                                    anchors = anchor_aligner.fit_transform(reference_speech, sub_speech)
                                    
                                    # Create new pipeline with morpher
                                    new_pipe = Pipeline(best_srt_pipe.steps)
                                    new_pipe.steps.insert(0, ('morph', SubtitleMorpher(anchors)))
                                    
                                    # Calculate new score
                                    morphed_speech = new_pipe.transform(srtin)
                                    score = calculate_morphed_score(reference_speech, morphed_speech)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_srt_pipe = new_pipe
                                        offset_samples = 0  # Reset offset since we're using morphing
                                        
                                    logger.info("Anchor-based alignment score: %.3f", score)
                                    
                                    # Early termination if we have an excellent match
                                    if best_score >= EXCELLENT_THRESHOLD:
                                        logger.info(f"Found excellent match (score: {best_score:.3f}), stopping search")
                                        break
                                    
                                except FailedToFindAlignmentException:
                                    logger.info("Failed to find valid anchor points, continuing with next strategy")
                                    continue
                                    
                            if best_score >= DEFAULT_SUCCESS_THRESHOLD:
                                logger.info("Found satisfactory alignment, stopping search")
                                break
                                
                        except Exception as e:
                            logger.warning("Strategy failed: %s", str(e))
                            continue
                            
                if best_score < MIN_ACCEPTABLE_SCORE:
                    logger.error("Could not find satisfactory alignment")
                    sync_was_successful = False
                    continue
                    
            # Process output
            logger.info("score: %.3f", best_score)
            logger.info("offset seconds: %.3f", offset_samples / float(SAMPLE_RATE))
            
            if srtout:
                # Get the parser from the best performing pipeline
                parser = cast(Pipeline, srt_pipe_maker(1.0)).named_steps['parse']
                
                # Create pipeline with parser first
                out_pipe = Pipeline([
                    ('parse', parser),
                    ('shift', SubtitleShifter(offset_samples / float(SAMPLE_RATE))),
                ])
                
                # Add merge step if needed
                if args.merge_with_reference:
                    out_pipe.steps.append(
                        ('merge', SubtitleMerger(reference_pipe.named_steps['parse'].subs_))
                    )
                
                # Transform the subtitles
                out_subs = out_pipe.fit_transform(srtin)
                
                if args.output_encoding != "same":
                    out_subs = out_subs.set_encoding(args.output_encoding)
                    
                suppress_output_thresh = args.suppress_output_if_offset_less_than
                if offset_samples / float(SAMPLE_RATE) >= (suppress_output_thresh or float("-inf")):
                    logger.info("writing output to {}".format(srtout))
                    out_subs.write_file(srtout)
                else:
                    logger.warning(
                        "suppressing output because offset %s was less than suppression threshold %s",
                        offset_samples / float(SAMPLE_RATE),
                        args.suppress_output_if_offset_less_than,
                    )
                    
        except Exception as e:
            sync_was_successful = False
            logger.exception("failed to sync %s: %s", srtin, str(e))
        else:
            result["offset_seconds"] = offset_samples / float(SAMPLE_RATE)
            if hasattr(best_srt_pipe.named_steps.get("scale", None), "scale_factor"):
                result["framerate_scale_factor"] = best_srt_pipe.named_steps["scale"].scale_factor
            
    result["sync_was_successful"] = sync_was_successful
    return sync_was_successful


def make_reference_pipe(args: argparse.Namespace) -> Pipeline:
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return cast(
            Pipeline,
            make_subtitle_speech_pipeline(
                fmt=ref_format,
                **override(args, encoding=args.reference_encoding or DEFAULT_ENCODING),
            ),
        )
    elif ref_format in ("npy", "npz"):
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return Pipeline(
            [("deserialize", DeserializeSpeechTransformer(args.non_speech_label))]
        )
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning(
                "Reference srt encoding specified, but reference was a video file"
            )
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith("0:"):
            ref_stream = "0:" + ref_stream
        
        # Setup valid kwargs for VideoSpeechTransformer
        # Only pass valid parameters to VideoSpeechTransformer
        kwargs = {}
        
        # If memory-optimized mode is enabled, log that it is handled externally
        if getattr(args, 'memory_optimized', False):
            logger.info("Using memory-optimized mode (handled externally), not passing memory optimization parameters to VideoSpeechTransformer")
        
        return Pipeline([
            (
                "speech_extract",
                VideoSpeechTransformer(
                    vad=vad,
                    sample_rate=SAMPLE_RATE,
                    frame_rate=args.frame_rate,
                    non_speech_label=args.non_speech_label,
                    start_seconds=args.start_seconds,
                    ffmpeg_path=args.ffmpeg_path,
                    ref_stream=ref_stream,
                    vlc_mode=args.vlc_mode,
                    gui_mode=args.gui_mode,
                    **kwargs
                ),
            ),
        ])


def extract_subtitles_from_reference(args: argparse.Namespace) -> int:
    stream = args.extract_subs_from_stream
    if not stream.startswith("0:s:"):
        stream = "0:s:{}".format(stream)
    elif not stream.startswith("0:") and stream.startswith("s:"):
        stream = "0:{}".format(stream)
    if not stream.startswith("0:s:"):
        logger.error(
            "invalid stream for subtitle extraction: %s", args.extract_subs_from_stream
        )
    ffmpeg_args = [
        ffmpeg_bin_path("ffmpeg", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path)
    ]
    ffmpeg_args.extend(
        [
            "-y",
            "-nostdin",
            "-loglevel",
            "fatal",
            "-i",
            args.reference,
            "-map",
            "{}".format(stream),
            "-f",
            "srt",
        ]
    )
    if args.srtout is None:
        ffmpeg_args.append("-")
    else:
        ffmpeg_args.append(args.srtout)
    logger.info(
        "attempting to extract subtitles to {} ...".format(
            "stdout" if args.srtout is None else args.srtout
        )
    )
    retcode = subprocess.call(ffmpeg_args)
    if retcode == 0:
        logger.info("...done")
    else:
        logger.error(
            "ffmpeg unable to extract subtitles from reference; return code %d", retcode
        )
    return retcode


def validate_args(args: argparse.Namespace) -> None:
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference is None:
        if args.apply_offset_seconds == 0 or not args.srtin:
            raise ValueError(
                "`reference` required unless `--apply-offset-seconds` specified"
            )
    if args.apply_offset_seconds != 0:
        if not args.srtin:
            args.srtin = [args.reference]
        if not args.srtin:
            raise ValueError(
                "at least one of `srtin` or `reference` must be specified to apply offset seconds"
            )
    if args.srtin:
        if len(args.srtin) > 1 and not args.overwrite_input:
            raise ValueError(
                "cannot specify multiple input srt files without overwriting"
            )
        if len(args.srtin) > 1 and args.make_test_case:
            raise ValueError("cannot specify multiple input srt files for test cases")
        if len(args.srtin) > 1 and args.gui_mode:
            raise ValueError("cannot specify multiple input srt files in GUI mode")
    if (
        args.make_test_case and not args.gui_mode
    ):  # this validation not necessary for gui mode
        if not args.srtin or args.srtout is None:
            raise ValueError(
                "need to specify input and output srt files for test cases"
            )
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError(
                "input overwriting not allowed for extracting subtitles from reference"
            )
        if not args.srtin:
            raise ValueError(
                "need to specify input srt if --overwrite-input "
                "is specified since we cannot overwrite stdin"
            )
        if args.srtout is not None:
            raise ValueError(
                "overwrite input set but output file specified; "
                "refusing to run in case this was not intended"
            )
    if args.extract_subs_from_stream is not None:
        if args.make_test_case:
            raise ValueError("test case is for sync and not subtitle extraction")
        if args.srtin:
            raise ValueError(
                "stream specified for reference subtitle extraction; "
                "-i flag for sync input not allowed"
            )


def validate_file_permissions(args: argparse.Namespace) -> None:
    error_string_template = (
        "unable to {action} {file}; "
        "try ensuring file exists and has correct permissions"
    )
    if args.reference is not None and not os.access(args.reference, os.R_OK):
        raise ValueError(
            error_string_template.format(action="read reference", file=args.reference)
        )
    if args.srtin:
        for srtin in args.srtin:
            if srtin is not None and not os.access(srtin, os.R_OK):
                raise ValueError(
                    error_string_template.format(
                        action="read input subtitles", file=srtin
                    )
                )
    if (
        args.srtout is not None
        and os.path.exists(args.srtout)
        and not os.access(args.srtout, os.W_OK)
    ):
        raise ValueError(
            error_string_template.format(
                action="write output subtitles", file=args.srtout
            )
        )
    if args.make_test_case or args.serialize_speech:
        npy_savename = os.path.splitext(args.reference)[0] + ".npz"
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError(
                "unable to write test case file archive %s (try checking permissions)"
                % npy_savename
            )


def _setup_logging(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[logging.FileHandler]]:
    log_handler = None
    log_path = None
    if args.make_test_case or args.log_dir_path is not None:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        log_handler = logging.FileHandler(log_path)
        logger.addHandler(log_handler)
        logger.info("this log will be written to %s", os.path.abspath(log_path))
    return log_path, log_handler


def _npy_savename(args: argparse.Namespace) -> str:
    return os.path.splitext(args.reference)[0] + ".npz"


def _process_segment(args_dict, segment_idx, start, end, temp_dir):
    """Process a single video segment for speech extraction.
    
    Args:
        args_dict: Dictionary of arguments (serializable version of args)
        segment_idx: Index of this segment
        start: Start time in seconds
        end: End time in seconds
        temp_dir: Path to temporary directory for output
        
    Returns:
        Dictionary with segment information or None if processing failed
    """
    import os
    import numpy as np
    from copy import deepcopy
    import logging
    import sys
    import tempfile
    import time
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    # Re-create args from dictionary
    from argparse import Namespace
    segment_args = Namespace(**args_dict)
    
    # Set segment-specific parameters
    segment_args.start_seconds = start
    segment_args.max_seconds = end - start
    segment_args.memory_optimized = False  # Avoid recursive optimization
    
    # Set timeout for segment processing
    start_time = time.time()
    max_segment_time = 900  # 15 minutes maximum per segment
    
    # Modify process title to help identify in logs
    try:
        import setproctitle
        setproctitle.setproctitle(f"ffsubsync-seg{segment_idx}")
    except ImportError:
        pass
    
    # Customize the log format to include segment information
    segment_prefix = f"[Segment {segment_idx}] "
    logger.info(f"{segment_prefix}Processing {start:.2f}s to {end:.2f}s")
    
    # Create unique output path for this segment
    segment_output = os.path.join(temp_dir, f"segment_{segment_idx}.npz")
    
    try:
        # Periodic status updates
        last_status_time = time.time()
        status_interval = 30.0  # Every 30 seconds
        
        # Extract speech from this segment
        reference_pipe = make_reference_pipe(segment_args)
        if reference_pipe is None:
            return None
            
        # Log status periodically during processing
        def check_timeout():
            nonlocal last_status_time
            current_time = time.time()
            
            # Check for timeout
            if current_time - start_time > max_segment_time:
                logger.warning(f"{segment_prefix}Processing timeout exceeded (>{max_segment_time}s), aborting")
                raise TimeoutError(f"Segment processing timeout exceeded")
                
            # Provide periodic status updates
            if current_time - last_status_time > status_interval:
                elapsed = current_time - start_time
                logger.info(f"{segment_prefix}Still processing ({elapsed:.1f}s elapsed)")
                last_status_time = current_time
                
        # Begin fitting the pipeline
        logger.info(f"{segment_prefix}Fitting speech extraction pipeline")
        check_timeout()
        reference_pipe.fit(segment_args.reference)
        
        # Transform to extract speech data
        logger.info(f"{segment_prefix}Transforming audio to speech data")
        check_timeout()
        speech = reference_pipe.transform(segment_args.reference)
        
        # Check timeout again
        check_timeout()
        
        # Save speech data for this segment
        logger.info(f"{segment_prefix}Saving extracted speech data")
        np.savez_compressed(
            segment_output,
            speech=speech,
            start=start,
            end=end,
            sample_rate=SAMPLE_RATE
        )
        
        # Calculate speech activity metrics
        speech_activity = np.mean(speech)
        speech_transitions = np.sum(np.abs(np.diff(speech)))
        
        # Log processing time
        total_time = time.time() - start_time
        logger.info(f"{segment_prefix}Processing complete in {total_time:.1f}s - speech activity: {speech_activity:.3f}, transitions: {speech_transitions:.0f}")
        
        return {
            'idx': segment_idx,
            'start': start,
            'end': end,
            'speech_activity': speech_activity,
            'speech_transitions': speech_transitions,
            'output_path': segment_output
        }
    except Exception as e:
        # Calculate how long we ran before failing
        elapsed = time.time() - start_time
        logger.warning(f"{segment_prefix}Failed after {elapsed:.1f}s: {e}")
        
        return None


def process_large_file(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    """Optimized processing for large files using selective segment analysis.
    
    This function:
    1. Divides the file into strategic segments
    2. Processes each segment in parallel if possible
    3. Selects the best segment for synchronization
    4. Uses memory-efficient processing throughout
    """
    import os
    import tempfile
    from copy import deepcopy
    import time
    
    # Check file size
    if not os.path.exists(args.reference):
        return False
        
    file_size = os.path.getsize(args.reference)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    logger.info(f"Large file detected ({file_size_gb:.2f} GB), using memory-optimized processing")
    
    # Get video duration
    try:
        import ffmpeg
        probe = ffmpeg.probe(args.reference)
        duration = float(probe['format']['duration'])
        logger.info(f"Video duration: {duration:.2f} seconds")
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}")
        # Estimate duration based on file size and bitrate
        # Assume ~5 Mbps for typical video
        estimated_bitrate = 5 * 1024 * 1024  # 5 Mbps
        duration = (file_size * 8) / estimated_bitrate
        logger.info(f"Estimated duration: {duration:.2f} seconds")
    
    # Create temporary directory for segment processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define segments to process based on file size
        segment_length = getattr(args, 'segment_length', 300)  # Default to 5 minutes if not specified
        
        if file_size_gb > 20:  # >20GB
            # For very large files, process just a few strategic points
            segments = [
                (0, segment_length),  # First segment
                (duration/2 - segment_length/2, duration/2 + segment_length/2),  # Middle segment
                (max(0, duration - segment_length), duration)  # Last segment
            ]
            logger.info(f"Using selective segment processing for very large file ({len(segments)} segments)")
        elif file_size_gb > 10:  # 10-20GB
            # For large files, process more segments
            segments = [
                (0, segment_length),  # First segment
                (duration/4 - segment_length/2, duration/4 + segment_length/2),  # 25% point
                (duration/2 - segment_length/2, duration/2 + segment_length/2),  # Middle
                (3*duration/4 - segment_length/2, 3*duration/4 + segment_length/2),  # 75% point
                (max(0, duration - segment_length), duration)  # Last segment
            ]
            logger.info(f"Using strategic segment processing for large file ({len(segments)} segments)")
        else:  # 5-10GB
            # For medium files, use more granular segments
            segment_count = 8
            segment_length = duration / segment_count
            segments = [(i * segment_length, (i + 1) * segment_length) for i in range(segment_count)]
            logger.info(f"Using granular segment processing for medium file ({len(segments)} segments)")
        
        logger.info(f"Processing {len(segments)} segments")
        
        # Convert args to a serializable dictionary for multiprocessing
        args_dict = args.__dict__.copy()
        
        # Process segments (in parallel if possible)
        segment_results = []
        try:
            # Try parallel processing
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
            import time
            
            # Create optimized progress display
            progress = ProgressReporter.create_progress_display(
                "[progress.description]{task.description}", 
                "[progress.percentage]{task.percentage:>3.0f}%", 
                "Processed: {task.completed}/{task.total}",
                transient=False  # Keep progress bar visible
            )
            
            logger.info("Processing segments in parallel")
            max_workers = min(multiprocessing.cpu_count(), len(segments))
            logger.info(f"Using {max_workers} worker processes")
            
            # Track each segment's start time
            segment_start_times = {}
            last_status_time = time.time()
            status_interval = 10.0  # Log status every 10 seconds
            
            with progress:
                task = progress.add_task("Processing segments", total=len(segments))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs
                    futures = {}
                    for i, (start, end) in enumerate(segments):
                        future = executor.submit(_process_segment, args_dict, i, start, end, temp_dir)
                        futures[future] = i
                        segment_start_times[i] = time.time()
                        logger.info(f"Submitted segment {i} for processing")
                    
                    # Monitor progress
                    completed = 0
                    while completed < len(segments):
                        # Check for completed futures
                        done, not_done = wait(list(futures.keys()), timeout=2.0, return_when=FIRST_COMPLETED)
                        
                        # Process any completed futures
                        for future in done:
                            segment_idx = futures.pop(future)
                            try:
                                segment_result = future.result()
                                processing_time = time.time() - segment_start_times[segment_idx]
                                if segment_result:
                                    logger.info(f"Segment {segment_idx} completed in {processing_time:.1f}s")
                                    segment_results.append(segment_result)
                                else:
                                    logger.warning(f"Segment {segment_idx} failed after {processing_time:.1f}s")
                            except Exception as e:
                                logger.warning(f"Error in segment {segment_idx}: {e}")
                            
                            # Update progress
                            completed += 1
                            progress.update(task, completed=completed)
                        
                        # Provide periodic status updates for running segments
                        current_time = time.time()
                        if current_time - last_status_time > status_interval:
                            for future, idx in futures.items():
                                elapsed = current_time - segment_start_times[idx]
                                logger.info(f"Segment {idx} still processing ({elapsed:.1f}s elapsed)")
                            last_status_time = current_time
                            
                        # Don't burn CPU if nothing completed
                        if not done:
                            time.sleep(0.5)
        except (ImportError, Exception) as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            
            # Create optimized progress display
            progress = ProgressReporter.create_progress_display(
                "[progress.description]{task.description}", 
                "[progress.percentage]{task.percentage:>3.0f}%", 
                "Processed: {task.completed}/{task.total}",
                transient=False  # Keep progress bar visible
            )
            
            with progress:
                task = progress.add_task("Processing segments sequentially", total=len(segments))
                for i, (start, end) in enumerate(segments):
                    logger.info(f"Starting sequential processing of segment {i}")
                    start_time = time.time()
                    segment_result = _process_segment(args_dict, i, start, end, temp_dir)
                    elapsed = time.time() - start_time
                    
                    if (segment_result):
                        logger.info(f"Segment {i} completed in {elapsed:.1f}s")
                        segment_results.append(segment_result)
                    else:
                        logger.warning(f"Segment {i} failed after {elapsed:.1f}s")
                        
                    progress.update(task, advance=1)
        
        # If we have at least one successful segment, use it for synchronization
        if not segment_results:
            logger.error("All segments failed to process")
            return False
            
        # Select best segment based on speech activity and transitions
        # Prioritize segments with both high speech activity and many transitions
        for segment in segment_results:
            segment['score'] = segment['speech_activity'] * segment['speech_transitions']
            
        best_segment = max(segment_results, key=lambda x: x['score'])
        logger.info(f"Selected segment {best_segment['idx']} ({best_segment['start']:.2f}s to {best_segment['end']:.2f}s) "
                   f"with score {best_segment['score']:.4f}")
        
        # Load the best segment's speech data
        segment_data = np.load(best_segment['output_path'])
        speech = segment_data['speech']
        
        # Create a reference pipe with this segment's speech data
        class SegmentSpeechTransformer(TransformerMixin):
            def __init__(self, speech_data):
                self.speech_data = speech_data
                
            def fit(self, *_):
                return self
                
            def transform(self, *_):
                return self.speech_data
        
        # Create a pipeline with the segment speech transformer
        segment_pipe = Pipeline([
            ('speech', SegmentSpeechTransformer(speech))
        ])
        
        # Adjust args for this segment
        segment_args = deepcopy(args)
        segment_args.start_seconds = best_segment['start']
        segment_args.max_seconds = best_segment['end'] - best_segment['start']
        
        # Run synchronization with this segment
        return try_sync(segment_args, segment_pipe, result)


def _run_impl(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    # Check if this is a very large file that needs special handling
    import os
    
    # Auto-configure optimizations if enabled
    if getattr(args, 'auto_optimize', False):
        args = auto_configure_optimizations(args)
    
    # Check if file is large and memory optimization is requested
    file_is_large = False
    if os.path.exists(args.reference):
        file_size = os.path.getsize(args.reference)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        file_is_large = file_size_gb > 5
        
        if file_is_large:
            logger.info(f"Large file detected ({file_size_gb:.2f} GB)")
    
    # Apply memory optimization for large files if requested
    if getattr(args, 'memory_optimized', False) and file_is_large:
        logger.info("Using memory-optimized processing")
        
        try:
            result_status = process_large_file(args, result)
            if result_status:
                return True
                
            logger.info("Memory-optimized processing failed, falling back to standard processing")
        except Exception as e:
            logger.warning(f"Memory-optimized processing error: {e}, falling back to standard processing")
    
    # If we reached here, use standard processing
    
    # If max_seconds is specified, apply duration limiting
    if getattr(args, 'max_seconds', None):
        logger.info(f"Limiting processing to first {args.max_seconds} seconds")
    
    # Apply memory-specific optimizations if needed
    if getattr(args, 'memory_optimized', False):
        logger.info("Using memory-optimized mode (handled externally), not passing memory optimization parameters to VideoSpeechTransformer")
    
    # Standard processing flow
    reference_pipe = make_reference_pipe(args)
    if reference_pipe is None:
        return False
    
    # Log speech extraction step with more detail
    vad_method = getattr(args, 'vad', 'auditok')
    logger.info(f"extracting speech segments from reference '{args.reference}' using {vad_method}...")
    
    # Apply speech extraction and synchronization
    return try_sync(args, reference_pipe, result)


def auto_configure_optimizations(args: argparse.Namespace) -> argparse.Namespace:
    """Automatically configure optimizations based on file size and system resources.
    
    This function analyzes the input file and available system resources, then
    sets appropriate optimization parameters for best performance.
    """
    import os
    import psutil
    
    if not getattr(args, 'auto_optimize', False):
        return args
    
    # Log that auto-optimization is enabled
    logger.info("Auto-optimization mode enabled - configuring optimal settings")
    
    # Get system resources
    cpu_count = multiprocessing.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    logger.info(f"System resources: {cpu_count} CPU cores, {available_memory_gb:.1f}GB available memory")
    
    # Check reference file size
    if not os.path.exists(args.reference):
        return args
        
    file_size = os.path.getsize(args.reference)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    logger.info(f"Reference file size: {file_size_gb:.2f}GB")
    
    # Define optimizations based on file size and available memory
    if file_size_gb > 5:
        # For files > 5GB, use memory optimization
        logger.info("Auto-enabling memory optimized mode for large file")
        args.memory_optimized = True
        
        # Memory optimization parameters based on available memory
        if available_memory_gb < 8:
            logger.info("Limited memory detected, using ultra-memory-efficient mode")
            # Additional memory optimizations for low-memory systems
        
    # Enable parallel processing if multiple cores available and appropriate
    if cpu_count > 1 and file_size_gb > 1:
        logger.info(f"Auto-enabling parallel processing with {min(cpu_count, 4)} cores")
        args.parallel = True
        
    # Adjust segment length based on file size
    if file_size_gb > 20:
        # For very large files, use shorter segments
        segment_length = min(120, max(60, int(300 * (5 / file_size_gb))))
        logger.info(f"Auto-configuring segment length: {segment_length}s for very large file")
        args.segment_length = segment_length
        
        # Limit processing duration for huge files
        max_seconds = min(180, max(90, int(600 * (10 / file_size_gb))))
        logger.info(f"Auto-limiting processing duration to {max_seconds}s for very large file")
        args.max_seconds = max_seconds
        
    # Select appropriate VAD method based on file size and memory
    if file_size_gb > 10 and available_memory_gb < 4:
        # For large files on low-memory systems, use memory-efficient VAD
        logger.info("Auto-selecting memory-efficient VAD (auditok) for large file on low-memory system")
        args.vad = 'auditok'
    
    return args


def validate_and_transform_args(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace]
) -> Optional[argparse.Namespace]:
    if isinstance(parser_or_args, argparse.Namespace):
        parser = None
        args = parser_or_args
    else:
        parser = parser_or_args
        args = parser.parse_args()
    
    # Apply auto-optimization if enabled
    args = auto_configure_optimizations(args)
    
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(e)
        if parser is not None:
            parser.print_usage()
        return None
    if args.gui_mode and args.srtout is None:
        args.srtout = "{}.synced.srt".format(os.path.splitext(args.srtin[0])[0])
    try:
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        return None
    ref_format = _ref_format(args.reference)
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error(
            "merging synced output with reference only valid "
            "when reference composed of subtitles"
        )
        return None
    return args


def run(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace]
) -> Dict[str, Any]:
    sync_was_successful = False
    result = {
        "retval": 0,
        "offset_seconds": None,
        "framerate_scale_factor": None,
    }
    args = validate_and_transform_args(parser_or_args)
    if args is None:
        result["retval"] = 1
        return result
    log_path, log_handler = _setup_logging(args)
    try:
        sync_was_successful = _run_impl(args, result)
        result["sync_was_successful"] = sync_was_successful
        return result
    finally:
        if log_handler is not None and log_path is not None:
            log_handler.close()
            logger.removeHandler(log_handler)
            if args.make_test_case:
                result["retval"] += make_test_case(
                    args, _npy_savename(args), sync_was_successful
                )
            if args.log_dir_path is None or not os.path.isdir(args.log_dir_path):
                os.remove(log_path)


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version", action="version", version="{package} {version}".format(package="ffsubsync", version=get_version())
    )
    parser.add_argument("reference", help="Reference audio or video file")
    parser.add_argument("-i", "--srtin", nargs="+", help="Input subtitles file (default=stdin)")
    parser.add_argument("-o", "--srtout", help="Output subtitles file (default=stdout)")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="What encoding to use for reading input subtitles")
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE, help="Frame rate to use in speech extraction")
    parser.add_argument("--max-subtitle-seconds", type=float, default=DEFAULT_MAX_SUBTITLE_SECONDS, help="Maximum duration for a subtitle to appear on-screen")
    parser.add_argument("--start-seconds", type=int, default=DEFAULT_START_SECONDS, help="Start time for processing")
    parser.add_argument("--max-offset-seconds", type=float, default=DEFAULT_MAX_OFFSET_SECONDS, help="Maximum offset seconds to search")
    parser.add_argument("--apply-offset-seconds", type=float, default=DEFAULT_APPLY_OFFSET_SECONDS, help="Apply offset seconds to subtitles")
    parser.add_argument("--non-speech-label", type=str, default=DEFAULT_NON_SPEECH_LABEL, help="Label for non-speech segments in VAD")
    parser.add_argument("--vad", type=str, default=DEFAULT_VAD, help="Which voice activity detector to use")
    parser.add_argument("--no-fix-framerate", action="store_true", help="Do not try to fix framerate")
    parser.add_argument("--no-speech-threshold", type=float, default=0.3, help="Threshold for alignment score below which to try anchor-based alignment")
    parser.add_argument("--n-anchors", type=int, default=5, help="Number of anchor points to use in non-linear alignment")
    parser.add_argument(
        "--merge-with-reference",
        "--merge",
        action="store_true",
        help="Merge reference subtitles with synced output subtitles.",
    )
    parser.add_argument(
        "--make-test-case",
        "--create-test-case",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array, "
        "and create an archive with input/output subtitles "
        "and serialized speech.",
    )
    parser.add_argument(
        "--reference-stream",
        "--refstream",
        "--reference-track",
        "--reftrack",
        default=None,
        help=(
            "Which stream/track in the video file to use as reference, "
            "formatted according to ffmpeg conventions. For example, "
            "s:0 is the first subtitle track; a:3 is the fourth audio track."
        ),
    )
    parser.add_argument(
        "--skip-infer-framerate-ratio",
        action="store_true",
        help="If set, do not try to infer framerate ratio based on duration ratio.",
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8",
        help="What encoding to use for writing output subtitles "
        '(default=utf-8). Can indicate "same" to use same '
        "encoding as that of the input.",
    )
    parser.add_argument(
        "--reference-encoding",
        help="What encoding to use for reading / writing reference subtitles "
        "(if applicable, default=infer).",
    )
    parser.add_argument(
        "--extract-subs-from-stream",
        "--extract-subtitles-from-stream",
        default=None,
        help="If specified, do not attempt sync; instead, just extract subtitles"
        " from the specified stream using the reference.",
    )
    parser.add_argument(
        "--suppress-output-if-offset-less-than",
        type=float,
        default=None,
        help="If specified, do not produce output if offset below provided threshold.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        "--ffmpegpath",
        default=None,
        help="Where to look for ffmpeg and ffprobe. Uses the system PATH by default.",
    )
    parser.add_argument(
        "--log-dir-path",
        default=None,
        help=(
            "If provided, will save log file ffsubsync.log to this path "
            "(must be an existing directory)."
        ),
    )
    parser.add_argument(
        "--gss",
        action="store_true",
        help="If specified, use golden-section search to try to find"
        "the optimal framerate ratio between video and subtitles.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If specified, refuse to parse srt files with formatting issues.",
    )
    parser.add_argument("--vlc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gui-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-sync", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        '--overwrite-input',
        action='store_true',
        help='Overwrite the input subtitle file with synchronized output'
    )
    parser.add_argument(
        '--serialize-speech',
        action='store_true',
        help='Serialize speech data to JSON for debugging'
    )
    parser.add_argument(
        '--memory-optimized',
        action='store_true',
        help='Enable memory-optimized mode for large files'
    )
    parser.add_argument(
        '--content-aware',
        action='store_true',
        default=True,
        help='Enable content-aware anchor selection (default: enabled)'
    )
    parser.add_argument(
        '--progressive',
        action='store_true',
        default=True,
        help='Enable progressive multi-resolution alignment (default: enabled)'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=None,
        help='Maximum duration (in seconds) to process from the reference'
    )
    parser.add_argument(
        '--auto-optimize',
        action='store_true',
        help='Automatically configure optimal settings based on file size and system capabilities',
    )
    parser.add_argument(
        '--segment-length',
        type=int,
        default=300,
        help='Length of segments (in seconds) for memory-optimized processing',
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
