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
                
                for strategy in alignment_strategies:
                    try:
                        if strategy['type'] == 'global':
                            # Initial linear alignment attempt
                            logger.info("Trying global alignment...")
                            (score, offset), pipe = MaxScoreAligner(
                                FFTAligner, srtin, SAMPLE_RATE, strategy['params']['max_offset_seconds']
                            ).fit_transform(reference_speech, srt_pipes)
                            
                            if score > best_score:
                                best_score = score
                                best_srt_pipe = pipe
                                offset_samples = offset
                                
                        elif strategy['type'] == 'anchors':
                            if best_score >= strategy['params'].get('min_score', 0):
                                logger.info("Previous alignment score sufficient, skipping anchor-based alignment")
                                continue
                                
                            logger.info("Trying anchor-based alignment with %s anchors...", 
                                      strategy['params']['n_anchors'])
                            
                            # Get speech signal from current best pipeline
                            sub_speech = best_srt_pipe.transform(srtin)
                            
                            # Find anchor points
                            anchor_aligner = AnchorAligner(
                                n_anchors=strategy['params']['n_anchors'],
                                max_anchors=strategy['params'].get('max_anchors', MAX_AUTO_ANCHORS)
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
                output_steps = []
                if offset_samples != 0:
                    offset_seconds = offset_samples / float(SAMPLE_RATE)
                    logger.info("offset seconds: %.3f", offset_seconds)
                    output_steps.append(("shift", SubtitleShifter(offset_seconds)))
                    
                if args.merge_with_reference:
                    output_steps.append(
                        ("merge", SubtitleMerger(reference_pipe.named_steps["parse"].subs_))
                    )
                    
                # Apply the pipeline steps
                out_pipe = Pipeline(output_steps) if output_steps else best_srt_pipe
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
        return Pipeline(
            [
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
                    ),
                ),
            ]
        )


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


def _run_impl(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    if args.extract_subs_from_stream is not None:
        result["retval"] = extract_subtitles_from_reference(args)
        return True
    if args.srtin is not None and (
        args.reference is None
        or (len(args.srtin) == 1 and args.srtin[0] == args.reference)
    ):
        return try_sync(args, None, result)
    reference_pipe = make_reference_pipe(args)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info("...done")
    if args.make_test_case or args.serialize_speech:
        logger.info("serializing speech...")
        np.savez_compressed(
            _npy_savename(args), speech=reference_pipe.transform(args.reference)
        )
        logger.info("...done")
        if not args.srtin:
            logger.info(
                "unsynchronized subtitle file not specified; skipping synchronization"
            )
            return False
    return try_sync(args, reference_pipe, result)


def validate_and_transform_args(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace]
) -> Optional[argparse.Namespace]:
    if isinstance(parser_or_args, argparse.Namespace):
        parser = None
        args = parser_or_args
    else:
        parser = parser_or_args
        args = parser.parse_args()
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
            "formatted according to ffmpeg conventions. For example, 0:s:0 "
            "uses the first subtitle track; 0:a:3 would use the third audio track. "
            "You can also drop the leading `0:`; i.e. use s:0 or a:3, respectively. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2`"
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
