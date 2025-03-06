# -*- coding: utf-8 -*-
import logging
import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.constants import (
    MIN_ACCEPTABLE_SCORE,
    MAX_AUTO_ANCHORS,
    MIN_ANCHOR_GAP_SECONDS,
    SAMPLE_RATE,
)


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


def downsample(signal: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a signal by taking every nth element."""
    return signal[::factor]


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None, window_size: int = 1000000, 
                 use_progressive: bool = True, downsample_factor: int = 10) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False
        self.window_size: int = window_size
        self.use_progressive: bool = use_progressive
        self.downsample_factor: int = downsample_factor

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

    def _sliding_window_fft(self, refstring: np.ndarray, substring: np.ndarray) -> Tuple[float, int]:
        best_score = float('-inf')
        best_offset = 0
        
        for i in range(0, len(refstring), self.window_size // 2):
            window = refstring[i:i + self.window_size]
            if len(window) < len(substring):
                continue  # Skip windows smaller than subtitle
            
            total_bits = math.log(len(substring) + len(window), 2)
            total_length = int(2 ** math.ceil(total_bits))
            extra_zeros = total_length - len(substring) - len(window)
            
            # More memory-efficient implementation
            padded_sub = np.zeros(total_length, dtype=np.float32)
            padded_sub[extra_zeros + len(window):extra_zeros + len(window) + len(substring)] = substring
            
            padded_ref = np.zeros(total_length, dtype=np.float32)
            padded_ref[:len(window)] = window[::-1]  # Flip in-place
            
            subft = np.fft.fft(padded_sub)
            refft = np.fft.fft(padded_ref)
            
            convolve = np.real(np.fft.ifft(subft * refft))
            convolve = self._eliminate_extreme_offsets_from_solutions(convolve, substring)
            
            window_best_idx = int(np.argmax(convolve))
            window_best_score = convolve[window_best_idx]
            window_best_offset = len(convolve) - 1 - window_best_idx - len(substring) + i
            
            if window_best_score > best_score:
                best_score = window_best_score
                best_offset = window_best_offset
        
        return best_score, best_offset
    
    def _progressive_fft(self, refstring: np.ndarray, substring: np.ndarray) -> Tuple[float, int]:
        """Progressive multi-resolution FFT alignment.
        
        First performs a coarse alignment on downsampled signals, then refines
        the result with a focused alignment on the full-resolution signals.
        """
        logger.info("Using progressive multi-resolution alignment")
        
        # Stage 1: Coarse alignment with downsampled signals
        factor = self.downsample_factor
        ds_refstring = downsample(refstring, factor)
        ds_substring = downsample(substring, factor)
        
        # Adjust max_offset_samples for downsampled signals
        original_max_offset = self.max_offset_samples
        if self.max_offset_samples is not None:
            self.max_offset_samples = self.max_offset_samples // factor
        
        # Perform coarse alignment
        coarse_score, coarse_offset = self._sliding_window_fft(ds_refstring, ds_substring)
        
        # Convert coarse offset to full resolution
        coarse_offset_full = coarse_offset * factor
        logger.info(f"Coarse alignment found offset: {coarse_offset_full} (score: {coarse_score:.3f})")
        
        # Restore original max_offset_samples
        self.max_offset_samples = original_max_offset
        
        # Stage 2: Refined alignment on a window around the coarse offset
        window_size = min(self.window_size, len(refstring) // 3)
        window_start = max(0, coarse_offset_full - window_size // 2)
        window_end = min(len(refstring), window_start + window_size)
        
        # Extract windows from full-resolution signals
        ref_window = refstring[window_start:window_end]
        
        # Adjust substring window based on the coarse offset
        sub_start = max(0, -coarse_offset_full)
        sub_end = min(len(substring), len(refstring) - coarse_offset_full)
        sub_window = substring[sub_start:sub_end]
        
        # Perform refined alignment on the windows
        refined_score, refined_offset = self._sliding_window_fft(ref_window, sub_window)
        
        # Combine coarse and refined offsets
        final_offset = coarse_offset_full + refined_offset - window_start
        logger.info(f"Refined alignment found offset: {final_offset} (score: {refined_score:.3f})")
        
        return refined_score, final_offset

    def fit(self, refstring, substring, get_score: bool = False) -> "FFTAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring]
        )
        
        # Use progressive approach for large files if enabled
        if self.use_progressive and len(refstring) > 100000:  # ~16 minutes at 100Hz
            self.best_score_, self.best_offset_ = self._progressive_fft(refstring, substring)
        elif len(refstring) + len(substring) > self.window_size:
            self.best_score_, self.best_offset_ = self._sliding_window_fft(refstring, substring)
        else:
            total_bits = math.log(len(substring) + len(refstring), 2)
            total_length = int(2 ** math.ceil(total_bits))
            extra_zeros = total_length - len(substring) - len(refstring)
            
            # More memory-efficient implementation
            padded_sub = np.zeros(total_length, dtype=np.float32)
            padded_sub[extra_zeros + len(refstring):extra_zeros + len(refstring) + len(substring)] = substring
            
            padded_ref = np.zeros(total_length, dtype=np.float32)
            padded_ref[:len(refstring)] = refstring[::-1]  # Flip in-place
            
            subft = np.fft.fft(padded_sub)
            refft = np.fft.fft(padded_ref)
            
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
        # Skip GSS if using anchors
        if hasattr(self, 'anchors_') and self.anchors_ and len(self.anchors_) > 2:
            return self
            
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


class AnchorAligner(TransformerMixin):
    def __init__(self, n_anchors='auto', max_anchors=MAX_AUTO_ANCHORS, min_anchor_gap=MIN_ANCHOR_GAP_SECONDS, 
                 window_size=30000, content_aware=True):
        """Initialize AnchorAligner.
        
        Args:
            n_anchors: Number of anchor points to find, or 'auto' for dynamic
            max_anchors: Maximum number of anchors when using auto mode
            min_anchor_gap: Minimum gap between anchors in seconds
            window_size: Size of window for local cross-correlation
            content_aware: Whether to use content-aware anchor selection
        """
        self.n_anchors = n_anchors
        self.max_anchors = max_anchors
        self.min_anchor_gap = min_anchor_gap
        self.window_size = window_size
        self.content_aware = content_aware
        self.anchors_ = None
        
    def _find_speech_transitions(self, speech: np.ndarray) -> np.ndarray:
        """Find regions with high speech transition density.
        
        Returns:
            Array of transition density scores for each position
        """
        # Calculate transitions (0->1 or 1->0)
        transitions = np.abs(np.diff(speech))
        
        # Compute transition density using a sliding window
        window_size = int(5 * SAMPLE_RATE)  # 5 second window
        transition_density = np.zeros_like(speech, dtype=float)
        
        # Use convolution for efficient sliding window sum
        if len(transitions) > 0:  # Ensure we have transitions to analyze
            density = np.convolve(transitions, np.ones(window_size), mode='same')
            transition_density[:-1] = density  # Adjust for diff length
            
        return transition_density
        
    def _find_distinctive_regions(self, speech: np.ndarray, n_regions=3) -> List[Tuple[int, int]]:
        """Find regions with distinctive speech patterns.
        
        Args:
            speech: Binary speech activity array
            n_regions: Number of distinctive regions to find
            
        Returns:
            List of (start_idx, end_idx) tuples for distinctive regions
        """
        # Find regions with high transition density
        transition_density = self._find_speech_transitions(speech)
        
        # Minimum region size (in samples)
        min_region_size = int(self.min_anchor_gap * SAMPLE_RATE)
        
        # Find peaks in transition density
        regions = []
        remaining_density = transition_density.copy()
        
        for _ in range(n_regions):
            if np.all(remaining_density == 0):
                break
                
            # Find region with highest density
            region_center = np.argmax(remaining_density)
            region_start = max(0, region_center - min_region_size // 2)
            region_end = min(len(speech), region_center + min_region_size // 2)
            
            # Add region if it has sufficient transitions
            if np.max(remaining_density[region_center]) > 0:
                regions.append((region_start, region_end))
                
            # Zero out this region to find next best
            remaining_density[region_start:region_end] = 0
            
        # Always include start and end regions if we don't have enough
        if len(regions) < n_regions:
            total_len = len(speech)
            
            # Add start region if not already included
            start_region = (0, min(total_len, min_region_size))
            if not any(r[0] == 0 for r in regions):
                regions.append(start_region)
                
            # Add end region if not already included
            end_region = (max(0, total_len - min_region_size), total_len)
            if not any(r[1] == total_len for r in regions):
                regions.append(end_region)
                
        return regions
        
    def _find_initial_anchors(self, ref_speech: np.ndarray, sub_speech: np.ndarray) -> List[Tuple[float, float]]:
        """Find initial anchor points at start and end."""
        total_len = len(ref_speech)
        anchors = [(0.0, 0.0)]  # Start point
        
        # Find end anchor using cross-correlation
        window_size = min(self.window_size, total_len // 4)
        ref_end = ref_speech[-window_size:]
        sub_end = sub_speech[-window_size:]
        correlation = signal.correlate(ref_end, sub_end, mode='full')
        lags = signal.correlation_lags(len(ref_end), len(sub_end), mode='full')
        peak_idx = np.argmax(correlation)
        lag = lags[peak_idx]
        
        # Add end anchor with proper time calculation
        end_time_ref = len(ref_speech)/SAMPLE_RATE
        end_time_sub = (len(sub_speech) + lag)/SAMPLE_RATE 
        anchors.append((end_time_ref, end_time_sub))
        
        # If content-aware is enabled, find additional distinctive regions
        if self.content_aware and self.n_anchors != 2:
            logger.info("Using content-aware anchor selection")
            distinctive_regions = self._find_distinctive_regions(ref_speech)
            
            for region_start, region_end in distinctive_regions:
                # Skip if region is too close to existing anchors
                region_time = (region_start + (region_end - region_start) / 2) / SAMPLE_RATE
                if any(abs(region_time - anchor[0]) < self.min_anchor_gap for anchor in anchors):
                    continue
                
                # Extract region from reference speech
                ref_region = ref_speech[region_start:region_end]
                
                # Find best match in subtitle speech using cross-correlation
                if len(ref_region) < len(sub_speech):
                    correlation = signal.correlate(sub_speech, ref_region, mode='valid')
                    match_idx = np.argmax(correlation)
                    match_score = correlation[match_idx] / (np.std(ref_region) * np.std(sub_speech[match_idx:match_idx+len(ref_region)]) * len(ref_region))
                    
                    if match_score > MIN_ACCEPTABLE_SCORE:
                        ref_time = region_start / SAMPLE_RATE
                        sub_time = match_idx / SAMPLE_RATE
                        anchors.append((ref_time, sub_time))
                        logger.info(f"Found distinctive anchor at {ref_time:.2f}s (score: {match_score:.3f})")
        
        return sorted(anchors, key=lambda x: x[0])
        
    def _find_midpoint_anchor(self, ref_speech: np.ndarray, sub_speech: np.ndarray, 
                            start_time: float, end_time: float) -> Optional[Tuple[float, float]]:
        """Find an anchor point between start_time and end_time."""
        total_len = len(ref_speech)
        mid_time = (start_time + end_time) / 2
        
        # Extract windows around midpoint
        mid_idx = int(mid_time * total_len)
        window_start = max(0, mid_idx - self.window_size // 2)
        window_end = min(total_len, mid_idx + self.window_size // 2)
        
        ref_window = ref_speech[window_start:window_end]
        sub_window = sub_speech[window_start:window_end]
        
        # Compute cross-correlation
        correlation = signal.correlate(ref_window, sub_window, mode='full')
        lags = signal.correlation_lags(len(ref_window), len(sub_window), mode='full')
        
        # Find peak
        peak_idx = np.argmax(correlation)
        lag = lags[peak_idx]
        score = correlation[peak_idx] / (np.std(ref_window) * np.std(sub_window) * len(ref_window))
        
        if score < MIN_ACCEPTABLE_SCORE:
            return None
            
        # Convert to timeline positions
        ref_time = mid_time
        sub_time = mid_time + float(lag) / total_len
        
        return (ref_time, sub_time)
        
    def _add_midpoint_anchors(self, anchors: List[Tuple[float, float]], 
                            ref_speech: np.ndarray, sub_speech: np.ndarray) -> List[Tuple[float, float]]:
        """Add additional anchor points between existing anchors."""
        if self.n_anchors != 'auto' and len(anchors) >= self.n_anchors:
            return anchors
            
        new_anchors = anchors.copy()
        added = True
        
        while added and len(new_anchors) < self.max_anchors:
            added = False
            current_anchors = sorted(new_anchors, key=lambda x: x[0])
            
            # If content-aware is enabled, prioritize regions with high transition density
            if self.content_aware:
                # Find gaps between anchors
                gaps = []
                for i in range(len(current_anchors) - 1):
                    start_time, end_time = current_anchors[i][0], current_anchors[i+1][0]
                    if end_time - start_time < self.min_anchor_gap:
                        continue
                    gaps.append((i, start_time, end_time))
                
                # Sort gaps by transition density
                gap_scores = []
                for i, start_time, end_time in gaps:
                    start_idx = int(start_time * SAMPLE_RATE)
                    end_idx = int(end_time * SAMPLE_RATE)
                    if start_idx >= end_idx or start_idx >= len(ref_speech) or end_idx > len(ref_speech):
                        gap_scores.append((i, 0))
                        continue
                    
                    # Calculate transition density in this gap
                    transitions = self._find_speech_transitions(ref_speech[start_idx:end_idx])
                    gap_score = np.mean(transitions) if len(transitions) > 0 else 0
                    gap_scores.append((i, gap_score))
                
                # Sort gaps by score (highest first)
                gap_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Try to add anchors in highest-scoring gaps first
                for i, _ in gap_scores:
                    start_time = current_anchors[i][0]
                    end_time = current_anchors[i+1][0]
                    
                    midpoint = self._find_midpoint_anchor(ref_speech, sub_speech, start_time, end_time)
                    if midpoint is not None:
                        new_anchors.append(midpoint)
                        added = True
                        break
            else:
                # Original approach: try gaps in order
                for i in range(len(current_anchors) - 1):
                    start_time, end_time = current_anchors[i][0], current_anchors[i+1][0]
                    if end_time - start_time < self.min_anchor_gap:
                        continue
                        
                    midpoint = self._find_midpoint_anchor(ref_speech, sub_speech, start_time, end_time)
                    if midpoint is not None:
                        new_anchors.append(midpoint)
                        added = True
                        break
                    
        return sorted(new_anchors, key=lambda x: x[0])
        
    def _anchors_are_plausible(self, anchors: List[Tuple[float, float]], total_duration: float) -> bool:
        """Check if the found anchors make sense."""
        if len(anchors) < 2:
            return False
            
        # Check time span
        times = [a[0] for a in anchors]
        if (max(times) - min(times)) * total_duration < self.min_anchor_gap:
            return False
            
        # Check for monotonicity and reasonable gaps
        for i in range(len(anchors) - 1):
            if anchors[i+1][1] <= anchors[i][1]:  # Must be strictly increasing
                return False
            if anchors[i+1][0] - anchors[i][0] < 1e-6:  # No zero-duration segments
                return False
                
        return True
        
    def fit(self, ref_speech: np.ndarray, sub_speech: np.ndarray) -> "AnchorAligner":
        """Find anchor points between reference and subtitle speech signals."""
        initial_anchors = self._find_initial_anchors(ref_speech, sub_speech)
        if not self._anchors_are_plausible(initial_anchors, len(ref_speech)):
            raise FailedToFindAlignmentException("Could not find plausible initial anchors")
            
        self.anchors_ = self._add_midpoint_anchors(initial_anchors, ref_speech, sub_speech)
        if not self._anchors_are_plausible(self.anchors_, len(ref_speech)):
            # Fall back to just initial anchors if midpoints aren't plausible
            self.anchors_ = initial_anchors
            
        return self
    
    def transform(self, *_) -> List[Tuple[float, float]]:
        """Return the found anchor points."""
        if self.anchors_ is None:
            raise ValueError("Must call fit before transform")
        return self.anchors_
