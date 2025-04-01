# -*- coding: utf-8 -*-
import logging
import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.constants import SAMPLE_RATE


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False

    def _eliminate_extreme_offsets_from_solutions(
        self, convolve: np.ndarray, substring: np.ndarray
    ) -> np.ndarray:
        convolve = np.copy(convolve)
        if self.max_offset_samples is None:
            return convolve

        def _offset_to_index(offset):
            return len(convolve) - 1 + offset - len(substring)

        convolve[: _offset_to_index(-self.max_offset_samples)] = float("-inf")
        convolve[_offset_to_index(self.max_offset_samples) :] = float("-inf")
        return convolve

    def _compute_argmax(self, convolve: np.ndarray, substring: np.ndarray) -> None:
        best_idx = int(np.argmax(convolve))
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]

    def fit(self, refstring, substring, get_score: bool = False) -> "FFTAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring]
        )
        total_bits = math.log(len(substring) + len(refstring), 2)
        total_length = int(2 ** math.ceil(total_bits))
        extra_zeros = total_length - len(substring) - len(refstring)
        subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(refstring)), substring))
        refft = np.fft.fft(
            np.flip(np.append(refstring, np.zeros(len(substring) + extra_zeros)), 0)
        )
        convolve = np.real(np.fft.ifft(subft * refft))
        self._compute_argmax(
            self._eliminate_extreme_offsets_from_solutions(convolve, substring),
            substring,
        )
        self.get_score_ = get_score
        return self

    def transform(self, *_) -> Union[int, Tuple[float, int]]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(
        self,
        base_aligner: Union[FFTAligner, Type[FFTAligner]],
        srtin: Optional[str] = None,
        sample_rate=None,
        max_offset_seconds=None,
    ) -> None:
        self.srtin: Optional[str] = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: Optional[int] = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(
                max_offset_samples=self.max_offset_samples
            )
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: Optional[int] = max_offset_seconds
        self._scores: List[Tuple[Tuple[float, int], Pipeline]] = []

    def fit_gss(self, refstring, subpipe_maker):
        def opt_func(framerate_ratio, is_last_iter):
            subpipe = subpipe_maker(framerate_ratio)
            substring = subpipe.fit_transform(self.srtin)
            score = self.base_aligner.fit_transform(
                refstring, substring, get_score=True
            )
            logger.info(
                "got score %.0f (offset %d) for ratio %.3f",
                score[0],
                score[1],
                framerate_ratio,
            )
            if is_last_iter:
                self._scores.append((score, subpipe))
            return -score[0]

        gss(opt_func, MIN_FRAMERATE_RATIO, MAX_FRAMERATE_RATIO)
        return self

    def fit(
        self, refstring, subpipes: Union[Pipeline, List[Pipeline]]
    ) -> "MaxScoreAligner":
        if not isinstance(subpipes, list):
            subpipes = [subpipes]
        for subpipe in subpipes:
            if callable(subpipe):
                self.fit_gss(refstring, subpipe)
                continue
            elif hasattr(subpipe, "transform"):
                substring = subpipe.transform(self.srtin)
            else:
                substring = subpipe
            self._scores.append(
                (
                    self.base_aligner.fit_transform(
                        refstring, substring, get_score=True
                    ),
                    subpipe,
                )
            )
        return self

    def transform(self, *_) -> Tuple[Tuple[float, float], Pipeline]:
        scores = self._scores
        if self.max_offset_samples is not None:
            scores = list(
                filter(lambda s: abs(s[0][1]) <= self.max_offset_samples, scores)
            )
        if len(scores) == 0:
            raise FailedToFindAlignmentException(
                "Synchronization failed; consider passing "
                "--max-offset-seconds with a number larger than "
                "{}".format(self.max_offset_seconds)
            )
        (score, offset), subpipe = max(scores, key=lambda x: x[0][0])
        return (score, offset), subpipe


# TODO: Implement confidence scoring
# TODO: Implement local scaling factor calculation? (More complex)
# TODO: Implement segment-level fallbacks based on confidence? (More complex)

# Confidence threshold for adding a new anchor (normalized score per sample)
# This value is empirical and might need tuning.
MIN_ANCHOR_CONFIDENCE_PER_SAMPLE = 0.15

class AnchorBasedAligner(TransformerMixin):
    def __init__(
        self,
        max_offset_seconds: float,
        max_anchors: int = 7,  # Max number of anchors including start/end
        min_segment_seconds: float = 5.0, # Minimum duration between anchors
        min_anchor_confidence: float = MIN_ANCHOR_CONFIDENCE_PER_SAMPLE,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.max_offset_seconds = max_offset_seconds
        self.max_anchors = max(max_anchors, 2) # Need at least start and end
        self.min_segment_samples = int(min_segment_seconds * sample_rate)
        self.min_anchor_confidence = min_anchor_confidence
        self.sample_rate = sample_rate
        self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        # Store timing map as: List[(ref_sample, sub_sample, confidence)]
        self.timing_map_: List[Tuple[int, int, float]] = []
        self.fft_aligner_ = FFTAligner(max_offset_samples=self.max_offset_samples)
        self.overall_confidence_: float = 0.0 # Average confidence of added anchors

    def _align_local_segment(
        self, ref_segment: np.ndarray, sub_search_area: np.ndarray
    ) -> Tuple[float, int, float]:
        """
        Aligns a reference segment within a subtitle search area.
        Returns: (raw_score, offset, normalized_score_per_sample)
        """
        if len(ref_segment) == 0 or len(sub_search_area) < len(ref_segment):
            # Cannot align if ref is empty or longer than search area
            return -float('inf'), 0, 0.0

        # Use FFTAligner (note: ref_segment is the 'substring' for FFTAligner here)
        raw_score, offset = self.fft_aligner_.fit(sub_search_area, ref_segment, get_score=True).transform()

        # Normalize score by length of the segment aligned
        # Raw score is max cross-correlation. Max possible score is len(ref_segment).
        # Normalized score is between -1 and 1 (ideally 0 to 1 for VAD)
        normalized_score = raw_score / len(ref_segment) if len(ref_segment) > 0 else 0.0

        # Note: The offset returned is where ref_segment starts within sub_search_area
        return raw_score, offset, normalized_score

    def fit(self, refstring: np.ndarray, substring: np.ndarray) -> "AnchorBasedAligner":
        """
        Fits the anchor-based alignment model.
        refstring, substring: VAD signals (numpy arrays of 0s and 1s or similar).
        """
        logger.info("Starting anchor-based alignment...")
        n_ref = len(refstring)
        n_sub = len(substring)

        # 1. Initial Global Alignment (Coarse)
        # We use this primarily to get a rough idea, maybe constrain search later
        global_score, global_offset_samples = self.fft_aligner_.fit(
            refstring, substring, get_score=True
        ).transform()
        logger.info(f"Initial global alignment: score={global_score:.2f}, offset={global_offset_samples / self.sample_rate:.3f}s")

        # Normalize global score for potential comparison later
        global_normalized_score = global_score / min(n_ref, n_sub) if min(n_ref, n_sub) > 0 else 0.0
        logger.info(f"Initial global alignment: score={global_score:.2f} (norm={global_normalized_score:.3f}), offset={global_offset_samples / self.sample_rate:.3f}s")

        # Initialize timing map with start and end points, adjusted by global offset
        # Assign initial confidence based on global normalized score? Or just 1.0? Let's use 1.0 for endpoints.
        initial_anchors = [
            (0, max(0, global_offset_samples), 1.0), # Start anchor (ref_sample, sub_sample, confidence)
            (n_ref - 1, max(0, n_sub - 1 + global_offset_samples), 1.0) # End anchor
        ]
        self.timing_map_ = sorted(initial_anchors, key=lambda x: x[0])
        logger.info(f"Initial timing map (based on global offset): {[(a[0]/self.sample_rate, a[1]/self.sample_rate, a[2]) for a in self.timing_map_]}")

        # 2. Iterative Anchor Insertion
        for _ in range(self.max_anchors - 2): # Start with 2 anchors (start/end)
            if len(self.timing_map_) >= self.max_anchors:
                break

            # Find the segment with the largest gap in reference time *and lowest confidence*?
            # For now, just largest gap meeting length constraint.
            best_segment_idx = -1
            max_ref_gap = -1
            for i in range(len(self.timing_map_) - 1):
                ref_start, _, _ = self.timing_map_[i]
                ref_end, _, _ = self.timing_map_[i+1]
                ref_gap = ref_end - ref_start
                # Ensure segment is large enough to be split
                if ref_gap > max_ref_gap and ref_gap >= 2 * self.min_segment_samples:
                    max_ref_gap = ref_gap
                    best_segment_idx = i

            if best_segment_idx == -1:
                logger.info("No further segments meet criteria for adding anchors.")
                break # No segment large enough to split further

            ref_start, sub_start, _ = self.timing_map_[best_segment_idx]
            ref_end, sub_end, _ = self.timing_map_[best_segment_idx+1]
            logger.info(f"Attempting to add anchor in segment {best_segment_idx} (Ref: {ref_start/self.sample_rate:.3f}s - {ref_end/self.sample_rate:.3f}s)")

            # --- Find Optimal Midpoint Anchor ---
            # Define search region (e.g., middle third of the segment)
            search_ref_start = ref_start + max_ref_gap // 3
            search_ref_end = ref_end - max_ref_gap // 3

            best_new_anchor_ref = -1
            best_new_anchor_sub = -1
            best_anchor_confidence = -float('inf')

            # Iterate through potential reference anchor points in the search region (middle third)
            # This is a simple scan; more sophisticated search (e.g., peak finding in correlation) could be better
            step = self.sample_rate // 10 # Check every 100ms (adjust as needed)
            for potential_ref_anchor in range(search_ref_start, search_ref_end, step):
                # Estimate corresponding subtitle point based on current linear interpolation between segment ends
                # Avoid division by zero if ref_end == ref_start (shouldn't happen if max_ref_gap > 0)
                if ref_end == ref_start: continue
                ref_ratio = (potential_ref_anchor - ref_start) / (ref_end - ref_start)
                potential_sub_anchor_est = int(sub_start + ref_ratio * (sub_end - sub_start))

                # Define local windows around the potential anchor for alignment check
                # Window size should be large enough for meaningful correlation but smaller than segment
                window_size = self.min_segment_samples // 2 # Example window size (tune?)

                local_ref_start = max(ref_start, potential_ref_anchor - window_size // 2)
                local_ref_end = min(ref_end, potential_ref_anchor + window_size // 2)
                # Estimate corresponding sub window start/end based on interpolation
                local_sub_start_est = int(sub_start + ((local_ref_start - ref_start) / (ref_end - ref_start)) * (sub_end - sub_start))
                local_sub_end_est = int(sub_start + ((local_ref_end - ref_start) / (ref_end - ref_start)) * (sub_end - sub_start))

                # Ensure calculated window is valid
                if local_ref_end <= local_ref_start or local_sub_end_est <= local_sub_start_est:
                    continue

                ref_segment_local = refstring[local_ref_start:local_ref_end]
                # Define the subtitle search area around the estimated subtitle window
                search_sub_start = max(0, local_sub_start_est - self.max_offset_samples)
                search_sub_end = min(n_sub, local_sub_end_est + self.max_offset_samples)

                # Ensure search area is valid and large enough
                if search_sub_end <= search_sub_start or len(ref_segment_local) == 0 or search_sub_end - search_sub_start < len(ref_segment_local):
                    continue
                sub_search_area = substring[search_sub_start:search_sub_end]

                # Align the local reference segment within the subtitle search area
                try:
                    raw_score, local_offset, normalized_score = self._align_local_segment(
                        ref_segment_local, sub_search_area
                    )

                    # Check if this potential anchor point is better than the current best
                    if normalized_score > best_anchor_confidence:
                        best_anchor_confidence = normalized_score
                        # The reference anchor point is the center of the local ref window
                        best_new_anchor_ref = local_ref_start + len(ref_segment_local) // 2
                        # The corresponding subtitle anchor point is the center of the aligned segment
                        aligned_sub_start = search_sub_start + local_offset
                        best_new_anchor_sub = aligned_sub_start + len(ref_segment_local) // 2

                except FailedToFindAlignmentException:
                     pass # Could not align this local segment, try next potential point

            # --- Add the best found anchor if confidence is sufficient ---
            if best_new_anchor_ref != -1 and best_anchor_confidence >= self.min_anchor_confidence:
                 # Check minimum segment length again with the new anchor
                 new_anchor = (best_new_anchor_ref, best_new_anchor_sub, best_anchor_confidence)
                 seg1_len = new_anchor[0] - ref_start
                 seg2_len = ref_end - new_anchor[0]
                 if seg1_len >= self.min_segment_samples and seg2_len >= self.min_segment_samples:
                     logger.info(f"Adding new anchor: Ref={new_anchor[0]/self.sample_rate:.3f}s, Sub={new_anchor[1]/self.sample_rate:.3f}s, Confidence={new_anchor[2]:.3f}")
                     self.timing_map_.insert(best_segment_idx + 1, new_anchor)
                     self.timing_map_ = sorted(self.timing_map_, key=lambda x: x[0]) # Keep sorted
                 else:
                     logger.info(f"Skipping potential anchor Ref={new_anchor[0]/self.sample_rate:.3f}s (Confidence={new_anchor[2]:.3f}) - violates min segment length.")
            elif best_new_anchor_ref != -1:
                 logger.info(f"Skipping potential anchor Ref={best_new_anchor_ref/self.sample_rate:.3f}s - confidence {best_anchor_confidence:.3f} below threshold {self.min_anchor_confidence:.3f}.")
            else:
                 logger.info(f"Could not find a suitable anchor point with sufficient confidence in segment {best_segment_idx}.")
                 # Optional: Mark this segment as un-splittable? For now, we just won't find anchors.

        # Calculate overall confidence (average of non-endpoint anchors)
        added_anchor_confidences = [a[2] for a in self.timing_map_[1:-1]] # Exclude start/end anchors
        self.overall_confidence_ = np.mean(added_anchor_confidences) if added_anchor_confidences else global_normalized_score

        logger.info(f"Final timing map ({len(self.timing_map_)} anchors): {[(a[0]/self.sample_rate, a[1]/self.sample_rate, f'{a[2]:.3f}') for a in self.timing_map_]}")
        logger.info(f"Overall anchor confidence score: {self.overall_confidence_:.3f}")
        return self

    def transform(self, *_) -> List[Tuple[int, int, float]]:
        """Returns the calculated timing map (including confidence scores)."""
        if not self.timing_map_:
             raise FailedToFindAlignmentException("Timing map could not be generated.")
        # Ensure map is sorted by reference time before returning
        return sorted(self.timing_map_, key=lambda x: x[0])

    def get_confidence(self) -> float:
        """Returns the overall confidence score of the alignment."""
        return self.overall_confidence_
