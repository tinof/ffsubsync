#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import json # Add json import
import subprocess
import sys
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ffsubsync.aligners import FFTAligner, MaxScoreAligner, AnchorBasedAligner, FailedToFindAlignmentException
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
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline,
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleMerger, SubtitleShifter, SubtitleWarper, SubsMixin
from ffsubsync.version import get_version


logger: logging.Logger = logging.getLogger(__name__)

# --- Auto-Mode Thresholds ---
# Thresholds for triggering fallback to anchor mode (empirical, may need tuning)
# Normalized score threshold (lower means potentially bad global alignment)
STANDARD_SCORE_THRESHOLD_NORM = 0.1
# Offset threshold in seconds (large offsets might indicate issues)
STANDARD_OFFSET_THRESHOLD_SEC = 60.0
# Confidence threshold for accepting the anchor mode result after fallback
ANCHOR_CONFIDENCE_FALLBACK_THRESHOLD = 0.2 # Slightly lower than adding individual anchors
# --- End Auto-Mode Thresholds ---


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
        # Check if log_path exists before copying
        if os.path.exists(log_path):
            shutil.copy(log_path, tar_dir)
        else:
            logger.warning(f"Log file {log_path} not found for test case.")
        # Check if srtin exists before copying
        if args.srtin and args.srtin[0] and os.path.exists(args.srtin[0]):
             shutil.copy(args.srtin[0], tar_dir)
        else:
             logger.warning(f"Input srt file {args.srtin[0]} not found for test case.")

        # Check if srtout exists before moving
        if sync_was_successful and args.srtout and os.path.exists(args.srtout):
            shutil.move(args.srtout, tar_dir)
        elif sync_was_successful:
             logger.warning(f"Output srt file {args.srtout} not found for test case (sync reported success).")

        ref_fmt = _ref_format(args.reference)
        if ref_fmt in SUBTITLE_EXTENSIONS:
            if os.path.exists(args.reference):
                shutil.copy(args.reference, tar_dir)
            else:
                 logger.warning(f"Reference subtitle file {args.reference} not found for test case.")
        elif args.serialize_speech or ref_fmt in ('npy', 'npz'):
             if os.path.exists(npy_savename):
                 shutil.copy(npy_savename, tar_dir)
             else:
                  logger.warning(f"Serialized speech file {npy_savename} not found for test case.")
        # No else needed for video reference as it's not copied

        supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
        preferred_formats = ["gztar", "bztar", "xztar", "zip", "tar"]
        archive_created = False
        for archive_format in preferred_formats:
            if archive_format in supported_formats:
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                archive_created = True
                break
        if not archive_created:
            logger.error(
                "failed to create test archive; no formats supported "
                "(this should not happen)"
            )
            return 1
        logger.info("...done")
    except Exception as e:
         logger.error(f"Error creating test case archive: {e}")
         return 1
    finally:
        if os.path.exists(tar_dir):
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


def try_sync(
    args: argparse.Namespace, reference_pipe: Optional[Pipeline], result: Dict[str, Any]
) -> Tuple[bool, Dict]: # Return success status and debug info
    # Initialize debug info dictionary
    debug_info = {
        "reference_file": args.reference,
        "subtitle_files": [],
    }
    # result["sync_was_successful"] = False # This will be set based on overall_sync_successful
    # sync_was_successful = True # This will be determined per file and aggregated
    # DEBUGGING: Log the state of args.srtin before the check
    logger.info(f"DEBUG: args.srtin before check: {args.srtin}")
    logger.info(
        "extracting speech segments from %s...",
        "stdin" if not args.srtin else "subtitles file(s) {}".format(args.srtin),
    )
    if not args.srtin:
        args.srtin = [None] # Handle stdin case

    all_files_successful = True # Track overall success across multiple files

    for srtin_path in args.srtin:
        # Per-file debug info and success status
        file_debug_info = {
            "subtitle_file": srtin_path or "stdin",
            "sync_successful": False,
            "mode_used": None, # 'standard', 'anchor_forced', 'anchor_fallback', 'skip'
            "standard_results": None,
            "anchor_results": None,
            "fallback_info": None,
            "final_decision": None,
            "error": None,
        }
        sync_successful_for_file = False # Reset for each file
        try:
            srtin = srtin_path # Use the actual path or None
            skip_sync = args.skip_sync or reference_pipe is None
            skip_infer_framerate_ratio = (
                args.skip_infer_framerate_ratio or reference_pipe is None
            )
            srtout = srtin if args.overwrite_input else args.srtout
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            # --- Prepare Reference Signal ---
            if reference_pipe is None and not skip_sync:
                 # Should only happen if called directly without reference (e.g., apply offset only)
                 raise ValueError("Reference pipeline is missing but sync was not skipped.")
            reference_signal = reference_pipe.transform(args.reference) if reference_pipe else np.array([])
            n_ref = len(reference_signal)

            # --- Prepare Original Subtitle Signal and Data ---
            # We always need the original (unscaled) subtitle signal and parsed subs
            original_srt_pipe = srt_pipe_maker(1.0)
            if callable(original_srt_pipe):
                raise ValueError("Subtitle pipe maker returned callable unexpectedly")
            original_srt_pipe.fit(srtin)
            original_subtitle_signal = original_srt_pipe.transform(srtin)
            original_subs_obj: SubsMixin = original_srt_pipe.named_steps["parse"].subs_
            n_sub = len(original_subtitle_signal)

            # --- Alignment Logic ---
            use_anchor_result = False
            timing_map = None
            standard_offset_seconds = 0.0
            standard_scale_factor = 1.0
            subs_to_warp = original_subs_obj # Default to original subs
            anchor_aligner = None # To access confidence later if needed
            standard_norm_score = 0.0 # Initialize
            standard_best_score = -float('inf') # Initialize
            anchor_confidence = 0.0 # Initialize

            if skip_sync:
                logger.warning("Sync skipped by user request.")
                file_debug_info["mode_used"] = "skip"
                timing_map = [(0, 0, 1.0)] # Dummy map for no sync (ref, sub, conf)
                use_anchor_result = True # Use warper with dummy map
                result["offset_seconds"] = 0.0 # Keep overall result dict updated
                result["framerate_scale_factor"] = 1.0
                sync_successful_for_file = True
                file_debug_info["final_decision"] = "Sync skipped by user"

            elif args.use_anchor_based_aligner:
                # --- Force Anchor Mode ---
                logger.info("Anchor mode explicitly selected.")
                file_debug_info["mode_used"] = "anchor_forced"
                try:
                    anchor_aligner = AnchorBasedAligner(
                        max_offset_seconds=args.max_offset_seconds,
                        sample_rate=SAMPLE_RATE
                        # TODO: Add args for max_anchors, min_segment_seconds, min_anchor_confidence
                    )
                    timing_map = anchor_aligner.fit(reference_signal, original_subtitle_signal).transform()
                    anchor_confidence = anchor_aligner.get_confidence()
                    file_debug_info["anchor_results"] = {
                        "confidence": anchor_confidence,
                        "num_anchors": len(timing_map),
                        "timing_map_sec": [(a[0]/SAMPLE_RATE, a[1]/SAMPLE_RATE, a[2]) for a in timing_map]
                    }
                    logger.info(f"Anchor-based alignment successful (Overall Confidence: {anchor_confidence:.3f}).")
                    if anchor_confidence < ANCHOR_CONFIDENCE_FALLBACK_THRESHOLD:
                         logger.warning("Anchor alignment confidence is low, result might be poor.")
                    use_anchor_result = True
                    result["offset_seconds"] = None # Keep overall result dict updated
                    result["framerate_scale_factor"] = None
                    sync_successful_for_file = True
                    file_debug_info["final_decision"] = "Used forced anchor result"
                except FailedToFindAlignmentException as e:
                    logger.error(f"Anchor-based alignment failed: {e}")
                    file_debug_info["anchor_results"] = {"error": str(e)}
                    file_debug_info["final_decision"] = "Forced anchor failed"
                    sync_successful_for_file = False
                    # No continue, let finally block handle debug log write

            else:
                # --- Standard Mode with Potential Fallback ---
                file_debug_info["mode_used"] = "standard" # Initial assumption
                logger.info("Attempting standard alignment...")
                standard_sync_success = False
                # standard_best_score = -float('inf') # Already initialized
                standard_offset_samples = 0
                best_srt_pipe = None

                try:
                    # Prepare potentially scaled subtitle pipes for standard mode
                    framerate_ratios = get_framerate_ratios_to_try(args)
                    srt_pipes = [original_srt_pipe] + [ # Reuse the already fitted original pipe
                        srt_pipe_maker(rat) for rat in framerate_ratios
                    ]
                    # Fit the scaled pipes (original is already fitted)
                    for srt_pipe in srt_pipes[1:]:
                        if callable(srt_pipe): continue
                        srt_pipe.fit(srtin)

                    # Add inferred framerate ratio pipe if applicable
                    if not skip_infer_framerate_ratio and hasattr(reference_pipe[-1], "num_frames"):
                        # Ensure num_frames exists and is valid
                        ref_num_frames = getattr(reference_pipe[-1], "num_frames", None)
                        sub_num_frames = getattr(original_srt_pipe[-1], "num_frames", None)
                        if ref_num_frames is not None and sub_num_frames is not None and sub_num_frames > 0:
                            inferred_framerate_ratio_from_length = float(ref_num_frames) / sub_num_frames
                            logger.info("inferred framerate ratio: %.3f" % inferred_framerate_ratio_from_length)
                            srt_pipes.append(
                                cast(Pipeline, srt_pipe_maker(inferred_framerate_ratio_from_length)).fit(srtin)
                            )
                        else:
                             logger.warning("Could not infer framerate ratio due to missing num_frames.")


                    # Run MaxScoreAligner
                    (standard_best_score, standard_offset_samples), best_srt_pipe = MaxScoreAligner(
                        FFTAligner, srtin, SAMPLE_RATE, args.max_offset_seconds
                    ).fit_transform(reference_signal, srt_pipes)

                    standard_offset_seconds = (
                        standard_offset_samples / float(SAMPLE_RATE) + args.apply_offset_seconds
                    )
                    scale_step = best_srt_pipe.named_steps["scale"]
                    standard_scale_factor = scale_step.scale_factor
                    subs_to_warp = scale_step.subs_ # Use potentially scaled subs for output if standard mode is chosen

                    # Normalize score by the length of the shorter signal
                    standard_norm_score = standard_best_score / min(n_ref, n_sub) if min(n_ref, n_sub) > 0 else 0.0

                    logger.info("Standard alignment results:")
                    logger.info(f"  Score: {standard_best_score:.3f} (norm={standard_norm_score:.3f})")
                    logger.info(f"  Offset: {standard_offset_seconds:.3f} seconds")
                    logger.info(f"  Framerate Scale Factor: {standard_scale_factor:.5f}")
                    result["offset_seconds"] = standard_offset_seconds # Keep overall result dict updated
                    result["framerate_scale_factor"] = standard_scale_factor
                    file_debug_info["standard_results"] = {
                        "score_raw": standard_best_score,
                        "score_normalized": standard_norm_score,
                        "offset_samples": standard_offset_samples,
                        "offset_seconds": standard_offset_seconds,
                        "scale_factor": standard_scale_factor,
                    }
                    standard_sync_success = True

                except FailedToFindAlignmentException as e:
                    logger.error(f"Standard alignment failed: {e}")
                    file_debug_info["standard_results"] = {"error": str(e)}
                    file_debug_info["final_decision"] = "Standard alignment failed"
                    sync_successful_for_file = False
                    # No continue, let finally block handle debug log write

                # --- Evaluate Standard Result and Decide Fallback ---
                if standard_sync_success:
                    trigger_fallback = False
                    fallback_reason = []
                    if standard_norm_score < STANDARD_SCORE_THRESHOLD_NORM:
                        reason = f"Standard score ({standard_norm_score:.3f}) < threshold ({STANDARD_SCORE_THRESHOLD_NORM:.3f})"
                        logger.warning(reason)
                        fallback_reason.append(reason)
                        trigger_fallback = True
                    if abs(standard_offset_seconds) > STANDARD_OFFSET_THRESHOLD_SEC:
                        reason = f"Standard offset ({standard_offset_seconds:.3f}s) > threshold ({STANDARD_OFFSET_THRESHOLD_SEC:.1f}s)"
                        logger.warning(reason)
                        fallback_reason.append(reason)
                        trigger_fallback = True

                    file_debug_info["fallback_info"] = {
                        "triggered": trigger_fallback,
                        "reason": fallback_reason,
                        "anchor_attempted": False,
                    }

                    if trigger_fallback:
                        file_debug_info["mode_used"] = "anchor_fallback" # Update mode
                        logger.warning("Attempting fallback to anchor-based alignment...")
                        file_debug_info["fallback_info"]["anchor_attempted"] = True
                        try:
                            anchor_aligner = AnchorBasedAligner(
                                max_offset_seconds=args.max_offset_seconds,
                                sample_rate=SAMPLE_RATE
                            )
                            timing_map = anchor_aligner.fit(reference_signal, original_subtitle_signal).transform()
                            anchor_confidence = anchor_aligner.get_confidence()
                            file_debug_info["anchor_results"] = {
                                "confidence": anchor_confidence,
                                "num_anchors": len(timing_map),
                                "timing_map_sec": [(a[0]/SAMPLE_RATE, a[1]/SAMPLE_RATE, a[2]) for a in timing_map]
                            }
                            logger.info(f"Anchor-based alignment fallback successful (Overall Confidence: {anchor_confidence:.3f}).")

                            if anchor_confidence >= ANCHOR_CONFIDENCE_FALLBACK_THRESHOLD:
                                decision = "Anchor fallback confidence sufficient. Using anchor result."
                                logger.info(decision)
                                use_anchor_result = True
                                subs_to_warp = original_subs_obj # Use original subs for warping
                                result["offset_seconds"] = None # Not applicable
                                result["framerate_scale_factor"] = None # Not applicable
                                sync_successful_for_file = True
                                file_debug_info["final_decision"] = decision
                            else:
                                decision = f"Anchor fallback confidence ({anchor_confidence:.3f}) below threshold ({ANCHOR_CONFIDENCE_FALLBACK_THRESHOLD:.3f}). Reverting to standard result."
                                logger.warning(decision)
                                use_anchor_result = False
                                sync_successful_for_file = True # Still use the standard result
                                file_debug_info["final_decision"] = decision
                        except FailedToFindAlignmentException as e:
                            decision = f"Anchor-based alignment fallback failed: {e}. Reverting to standard result."
                            logger.error(decision)
                            file_debug_info["anchor_results"] = {"error": str(e)}
                            use_anchor_result = False
                            sync_successful_for_file = True # Still use the standard result
                            file_debug_info["final_decision"] = decision
                    else:
                        decision = "Standard alignment result acceptable. Using standard result."
                        logger.info(decision)
                        use_anchor_result = False
                        sync_successful_for_file = True
                        file_debug_info["final_decision"] = decision

            # --- Apply Transformation (if sync was deemed successful by either method) ---
            if sync_successful_for_file:
                file_debug_info["sync_successful"] = True
                output_steps: List[Tuple[str, TransformerMixin]] = []
                if use_anchor_result:
                    logger.info("Applying non-linear warp based on timing map.")
                    # Use SubtitleWarper
                    output_steps.append(("warp", SubtitleWarper(timing_map, SAMPLE_RATE)))
                else:
                    logger.info("Applying global shift/scale based on standard alignment.")
                    # Use SubtitleShifter for standard mode
                    output_steps.append(("shift", SubtitleShifter(standard_offset_seconds)))

                if args.merge_with_reference and reference_pipe is not None and "parse" in reference_pipe.named_steps:
                     output_steps.append(
                         ("merge", SubtitleMerger(reference_pipe.named_steps["parse"].subs_))
                     )
                elif args.merge_with_reference:
                     logger.warning("Cannot merge with reference as reference pipe or parse step is missing.")


                output_pipe = Pipeline(output_steps)
                # Fit the output pipe using the appropriate subtitle object
                # (original for anchor mode, potentially scaled for standard mode)
                out_subs = output_pipe.fit_transform(subs_to_warp)

                if args.output_encoding != "same":
                    out_subs = out_subs.set_encoding(args.output_encoding)

                # Output suppression logic (only applicable for standard mode's global offset)
                suppress_output = False
                if not use_anchor_result:
                    suppress_output_thresh = args.suppress_output_if_offset_less_than
                    if standard_offset_seconds < (suppress_output_thresh or float("-inf")):
                        suppress_output = True
                        logger.warning(
                            "suppressing output because standard offset %.3f was less than suppression threshold %.3f",
                            standard_offset_seconds,
                            args.suppress_output_if_offset_less_than,
                        )

                if not suppress_output:
                    logger.info("writing output to {}".format(srtout or "stdout"))
                    out_subs.write_file(srtout)

        except Exception as e:
            sync_successful_for_file = False
            # Log general exceptions that might occur outside alignment itself
            logger.exception(f"Error during sync process for {srtin}: {e}")
            file_debug_info["error"] = f"Exception during sync: {e}"
            file_debug_info["final_decision"] = "Error during sync process"
        finally:
            # Update overall success tracker
            if not sync_successful_for_file:
                all_files_successful = False
            # Append this file's debug info regardless of success/failure
            debug_info["subtitle_files"].append(file_debug_info)

    # Aggregate overall success
    # overall_sync_successful = all(f.get("sync_successful", False) for f in debug_info["subtitle_files"])
    result["sync_was_successful"] = all_files_successful

    # Write debug log if path provided
    if args.debug_log:
        try:
            logger.info(f"Writing debug log to {args.debug_log}")
            # Create directory if it doesn't exist
            debug_dir = os.path.dirname(args.debug_log)
            if debug_dir and not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            with open(args.debug_log, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
        except Exception as log_e:
            logger.error(f"Failed to write debug log to {args.debug_log}: {log_e}")

    return all_files_successful


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
        ffmpeg_bin_path("ffmpeg", ffmpeg_resources_path=args.ffmpeg_path)
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
    if args.make_test_case:
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
    # Check write permission for debug log directory if specified
    if args.debug_log:
        debug_dir = os.path.dirname(args.debug_log)
        if debug_dir and not os.path.exists(debug_dir):
            # Check if we can create the directory
            try:
                os.makedirs(debug_dir, exist_ok=True)
                # Clean up if we just created it for the check
                if not os.listdir(debug_dir): # Check if empty
                     os.rmdir(debug_dir)
                else: # If not empty, maybe it existed partially? Check parent write perm
                     parent_dir = os.path.dirname(debug_dir)
                     if parent_dir and not os.access(parent_dir, os.W_OK):
                          raise PermissionError
            except Exception:
                 raise ValueError(
                      error_string_template.format(action="write debug log to directory", file=debug_dir)
                 )
        elif debug_dir and not os.access(debug_dir, os.W_OK):
             raise ValueError(
                  error_string_template.format(action="write debug log to directory", file=debug_dir)
             )
        elif not debug_dir and not os.access('.', os.W_OK): # Current directory if no path specified
             raise ValueError(
                  error_string_template.format(action="write debug log to current directory", file='.')
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
        # Ensure we can write to the log file path
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_handler = logging.FileHandler(log_path, mode='w') # Overwrite log each run
            logger.addHandler(log_handler)
            logger.info("This log will be written to %s", os.path.abspath(log_path))
        except Exception as e:
            logger.error(f"Could not configure file logging to {log_path}: {e}")
            log_handler = None # Ensure it's None if setup failed
            log_path = None
    return log_path, log_handler


def _npy_savename(args: argparse.Namespace) -> str:
    return os.path.splitext(args.reference)[0] + ".npz"


def _run_impl(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    if args.extract_subs_from_stream is not None:
        result["retval"] = extract_subtitles_from_reference(args)
        return True # Extraction is a separate task, assume success if no error
    if args.srtin is not None and (
        args.reference is None
        or (len(args.srtin) == 1 and args.srtin[0] == args.reference)
    ):
        # Handle case where only offset is applied or input is same as reference
        # Need to decide if try_sync should handle this or if it's an error
        # For now, let try_sync handle it (it will skip sync if reference_pipe is None)
        # try_sync now returns only bool
        return try_sync(args, None, result)

    reference_pipe = make_reference_pipe(args)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info("...done")
    if args.make_test_case or args.serialize_speech:
        logger.info("serializing speech...")
        try:
            np.savez_compressed(
                _npy_savename(args), speech=reference_pipe.transform(args.reference)
            )
            logger.info("...done")
        except Exception as e:
             logger.error(f"Failed to serialize speech: {e}")
             # Decide if this should be a fatal error? For now, just warn.
        if not args.srtin:
            logger.info(
                "Unsynchronized subtitle file not specified; skipping synchronization"
            )
            return False # Cannot sync without input subs
    # try_sync now returns only bool
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
        # Move permission validation here, after args are parsed
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        if parser is not None:
            parser.print_usage()
        return None
    except PermissionError as e: # Catch permission errors specifically
         logger.error(e)
         return None
    # try:
    #     validate_file_permissions(args)
    # except ValueError as e:
    #     logger.error(e)
    #     return None
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
    # Initialize result dict earlier
    result = {
        "retval": 0,
        "offset_seconds": None,
        "framerate_scale_factor": None,
        "sync_was_successful": False, # Default to False
        "error": None,
    }
    sync_was_successful = False # Local variable for overall success

    args = validate_and_transform_args(parser_or_args)
    if args is None:
        result["retval"] = 1
        return result

    log_path, log_handler = _setup_logging(args)
    try:
        # _run_impl now returns only the overall success status
        sync_was_successful = _run_impl(args, result)
        # result dictionary is populated within _run_impl and try_sync
        result["sync_was_successful"] = sync_was_successful
        if not sync_was_successful:
             result["retval"] = 1 # Indicate failure if any file failed
        return result
    except Exception as e:
         logger.exception(f"Unhandled exception during run: {e}")
         result["retval"] = 1
         result["error"] = str(e)
         result["sync_was_successful"] = False # Ensure failure state
         return result
    finally:
        # Close main log handler if it was created
        if log_handler is not None: # Check if handler was successfully created
            log_handler.close()
            logger.removeHandler(log_handler)
            # Handle test case creation (should happen after debug log potentially written)
            # Pass the final aggregated sync_was_successful status
            if args.make_test_case:
                # Ensure npy_savename is generated correctly
                npy_path = ""
                if args.reference and _ref_format(args.reference) not in SUBTITLE_EXTENSIONS:
                     npy_path = _npy_savename(args)
                elif args.reference and _ref_format(args.reference) in ('npy', 'npz'):
                     npy_path = args.reference

                test_case_ret = make_test_case(
                    args, npy_path, sync_was_successful
                )
                # Only add to retval if make_test_case failed (returned non-zero)
                if test_case_ret != 0:
                     result["retval"] = result.get("retval", 0) + test_case_ret

            # Remove the log file only if it wasn't meant to be kept
            if log_path and (args.log_dir_path is None or not os.path.isdir(args.log_dir_path)):
                 try:
                      if os.path.exists(log_path):
                           os.remove(log_path)
                 except Exception as rm_e:
                      logger.error(f"Failed to remove temporary log file {log_path}: {rm_e}")


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "reference",
        nargs="?",
        help=(
            "Reference (video, subtitles, or a numpy array with VAD speech) "
            "to which to synchronize input subtitles."
        ),
    )
    parser.add_argument(
        "-i", "--srtin", nargs="*", help="Input subtitles file (default=stdin)."
    )
    parser.add_argument(
        "-o", "--srtout", help="Output subtitles file (default=stdout)."
    )
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


def add_cli_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="{package} {version}".format(
            package=__package__, version=get_version()
        ),
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help=(
            "If specified, will overwrite the input srt "
            "instead of writing the output to a new file."
        ),
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="What encoding to use for reading input subtitles "
        "(default=%s)." % DEFAULT_ENCODING,
    )
    parser.add_argument(
        "--max-subtitle-seconds",
        type=float,
        default=DEFAULT_MAX_SUBTITLE_SECONDS,
        help="Maximum duration for a subtitle to appear on-screen "
        "(default=%.3f seconds)." % DEFAULT_MAX_SUBTITLE_SECONDS,
    )
    parser.add_argument(
        "--start-seconds",
        type=int,
        default=DEFAULT_START_SECONDS,
        help="Start time for processing "
        "(default=%d seconds)." % DEFAULT_START_SECONDS,
    )
    parser.add_argument(
        "--max-offset-seconds",
        type=float,
        default=DEFAULT_MAX_OFFSET_SECONDS,
        help="The max allowed offset seconds for any subtitle segment "
        "(default=%d seconds)." % DEFAULT_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--apply-offset-seconds",
        type=float,
        default=DEFAULT_APPLY_OFFSET_SECONDS,
        help="Apply a predefined offset in seconds to all subtitle segments "
        "(default=%d seconds)." % DEFAULT_APPLY_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help="Frame rate for audio extraction (default=%d)." % DEFAULT_FRAME_RATE,
    )
    parser.add_argument(
        "--skip-infer-framerate-ratio",
        action="store_true",
        help="If set, do not try to infer framerate ratio based on duration ratio.",
    )
    parser.add_argument(
        "--non-speech-label",
        type=float,
        default=DEFAULT_NON_SPEECH_LABEL,
        help="Label to use for frames detected as non-speech (default=%f)"
        % DEFAULT_NON_SPEECH_LABEL,
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
        "--vad",
        choices=[
            "subs_then_webrtc",
            "webrtc",
            "subs_then_auditok",
            "auditok",
            "subs_then_silero",
            "silero",
        ],
        default=None,
        help="Which voice activity detector to use for speech extraction "
        "(if using video / audio as a reference, default={}).".format(DEFAULT_VAD),
    )
    parser.add_argument(
        "--no-fix-framerate",
        action="store_true",
        help="If specified, subsync will not attempt to correct a framerate "
        "mismatch between reference and subtitles.",
    )
    parser.add_argument(
        "--serialize-speech",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array.",
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
    parser.add_argument("--skip-sync", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-anchor-based-aligner",
        "--anchor-mode",
        action="store_true",
        help="Use the experimental anchor-based aligner for non-linear synchronization.",
    )
    parser.add_argument(
        "--debug-log",
        default=None,
        metavar="FILEPATH",
        help="If provided, write detailed alignment metrics and decisions to this file (JSON format).",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
