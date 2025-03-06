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
)


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None, window_size: int = 1000000) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False
        self.window_size: int = window_size

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
            
            subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(window)), substring))
            refft = np.fft.fft(np.flip(np.append(window, np.zeros(len(substring) + extra_zeros)), 0))
            
            convolve = np.real(np.fft.ifft(subft * refft))
            convolve = self._eliminate_extreme_offsets_from_solutions(convolve, substring)
            
            window_best_idx = int(np.argmax(convolve))
            window_best_score = convolve[window_best_idx]
            window_best_offset = len(convolve) - 1 - window_best_idx - len(substring) + i
            
            if window_best_score > best_score:
                best_score = window_best_score
                best_offset = window_best_offset
        
        return best_score, best_offset

    def fit(self, refstring, substring, get_score: bool = False) -> "FFTAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring]
        )
        
        if len(refstring) + len(substring) > self.window_size:
            self.best_score_, self.best_offset_ = self._sliding_window_fft(refstring, substring)
        else:
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
    def __init__(self, n_anchors='auto', max_anchors=MAX_AUTO_ANCHORS, min_anchor_gap=MIN_ANCHOR_GAP_SECONDS, window_size=30000):
        """Initialize AnchorAligner.
        
        Args:
            n_anchors: Number of anchor points to find, or 'auto' for dynamic
            max_anchors: Maximum number of anchors when using auto mode
            min_anchor_gap: Minimum gap between anchors in seconds
            window_size: Size of window for local cross-correlation
        """
        self.n_anchors = n_anchors
        self.max_anchors = max_anchors
        self.min_anchor_gap = min_anchor_gap
        self.window_size = window_size
        self.anchors_ = None
        
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
        return anchors
        
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
