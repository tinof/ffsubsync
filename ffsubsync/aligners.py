import logging
import math
from collections import defaultdict
from typing import Optional, Union

import numpy as np

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin

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
        refstring, substring = (
            2 * np.array(s).astype(float) - 1 for s in [refstring, substring]
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

    def transform(self, *_) -> Union[int, tuple[float, int]]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class SegmentedAligner(TransformerMixin):
    """
    Segmented voting aligner for robust synchronization.

    Divides the reference and subtitle strings into overlapping windows,
    aligns each window independently using FFTAligner, and uses majority
    voting to find the consensus offset. This approach is more robust
    against false positives (e.g., intro music) that can mislead global
    alignment.
    """

    def __init__(
        self,
        max_offset_samples: Optional[int] = None,
        window_size_seconds: int = 600,  # 10 minutes default
        overlap_seconds: int = 300,  # 5 minutes overlap
        sample_rate: int = 100,  # From constants.SAMPLE_RATE
        tolerance_seconds: float = 0.5,  # Bin tolerance for voting
    ) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.window_size_seconds: int = window_size_seconds
        self.overlap_seconds: int = overlap_seconds
        self.sample_rate: int = sample_rate
        self.tolerance_seconds: float = tolerance_seconds
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False

    def fit(self, refstring, substring, get_score: bool = False) -> "SegmentedAligner":
        # Convert to numpy arrays (same as FFTAligner)
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = (
            2 * np.array(s).astype(float) - 1 for s in [refstring, substring]
        )

        # Calculate window parameters in samples
        window_size_samples = int(self.window_size_seconds * self.sample_rate)
        step_size_samples = int(
            (self.window_size_seconds - self.overlap_seconds) * self.sample_rate
        )
        tolerance_samples = int(self.tolerance_seconds * self.sample_rate)

        # Handle short inputs: fall back to single FFTAligner
        if len(refstring) < window_size_samples:
            logger.info(
                "Input too short for segmented alignment (%d < %d samples), "
                "falling back to standard FFTAligner",
                len(refstring),
                window_size_samples,
            )
            fft_aligner = FFTAligner(max_offset_samples=self.max_offset_samples)
            fft_aligner.fit(refstring, substring, get_score=get_score)
            self.best_offset_ = fft_aligner.best_offset_
            self.best_score_ = fft_aligner.best_score_
            self.get_score_ = get_score
            return self

        # Collect (offset, score) for each window
        window_results: list[tuple[int, float]] = []

        logger.info(
            "Running segmented alignment: window_size=%ds, overlap=%ds, tolerance=%.1fs",
            self.window_size_seconds,
            self.overlap_seconds,
            self.tolerance_seconds,
        )

        # Slide window across the reference and subtitle strings
        num_windows = 0
        for start_idx in range(
            0, len(refstring) - window_size_samples + 1, step_size_samples
        ):
            end_idx = min(start_idx + window_size_samples, len(refstring))

            # Extract windows
            ref_window = refstring[start_idx:end_idx]
            sub_window = (
                substring[start_idx:end_idx]
                if end_idx <= len(substring)
                else substring[start_idx:]
            )

            # Skip if subtitle window is too short
            if len(sub_window) < window_size_samples // 2:
                continue

            # Align this window
            try:
                aligner = FFTAligner(max_offset_samples=self.max_offset_samples)
                aligner.fit(ref_window, sub_window, get_score=True)

                if aligner.best_offset_ is not None and aligner.best_score_ is not None:
                    # Adjust offset to be relative to the global start (not window start)
                    global_offset = aligner.best_offset_  # Already relative to window
                    window_results.append((global_offset, aligner.best_score_))
                    num_windows += 1

            except Exception as e:
                logger.warning(f"Failed to align window {num_windows}: {e}")
                continue

        if len(window_results) == 0:
            raise FailedToFindAlignmentException(
                "Segmented alignment failed: no windows could be aligned"
            )

        logger.info(f"Aligned {num_windows} windows, aggregating results...")

        # Bin offsets by tolerance and vote
        offset_bins: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)

        for offset, score in window_results:
            # Find bin center (round to nearest tolerance_samples)
            bin_center = round(offset / tolerance_samples) * tolerance_samples
            offset_bins[bin_center].append((offset, score))

        # Vote: count votes per bin
        bin_votes: dict[int, int] = {
            bin_center: len(results) for bin_center, results in offset_bins.items()
        }

        # Find bin(s) with maximum votes
        max_votes = max(bin_votes.values())
        winning_bins = [
            bin_center for bin_center, votes in bin_votes.items() if votes == max_votes
        ]

        logger.info(
            "Voting results: %d bins, max_votes=%d, winning_bins=%s",
            len(bin_votes),
            max_votes,
            winning_bins,
        )

        # If tie, pick bin with highest cumulative score
        if len(winning_bins) > 1:
            bin_scores = {
                bin_center: sum(score for _, score in offset_bins[bin_center])
                for bin_center in winning_bins
            }
            winning_bin = max(bin_scores, key=bin_scores.get)
            logger.info(
                "Tie broken by score: bin %d selected with cumulative score %.0f",
                winning_bin,
                bin_scores[winning_bin],
            )
        else:
            winning_bin = winning_bins[0]

        # Aggregate offset and score from winning bin
        winning_results = offset_bins[winning_bin]
        self.best_offset_ = int(np.mean([offset for offset, _ in winning_results]))
        self.best_score_ = np.mean([score for _, score in winning_results])

        logger.info(
            "Consensus: offset=%d samples (%.2fs), score=%.0f from %d/%d windows",
            self.best_offset_,
            self.best_offset_ / self.sample_rate,
            self.best_score_,
            len(winning_results),
            num_windows,
        )

        self.get_score_ = get_score
        return self

    def transform(self, *_) -> Union[int, tuple[float, int]]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(
        self,
        base_aligner: Union[FFTAligner, type[FFTAligner]],
        srtin: Optional[str] = None,
        sample_rate=None,
        max_offset_seconds=None,
        **aligner_kwargs,
    ) -> None:
        self.srtin: Optional[str] = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: Optional[int] = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(
                max_offset_samples=self.max_offset_samples, **aligner_kwargs
            )
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: Optional[int] = max_offset_seconds
        self._scores: list[tuple[tuple[float, int], Pipeline]] = []

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
        self, refstring, subpipes: Union[Pipeline, list[Pipeline]]
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

    def transform(self, *_) -> tuple[tuple[float, float], Pipeline]:
        scores = self._scores
        if self.max_offset_samples is not None:
            scores = list(
                filter(lambda s: abs(s[0][1]) <= self.max_offset_samples, scores)
            )
        if len(scores) == 0:
            raise FailedToFindAlignmentException(
                "Synchronization failed; consider passing "
                "--max-offset-seconds with a number larger than "
                f"{self.max_offset_seconds}"
            )
        (score, offset), subpipe = max(scores, key=lambda x: x[0][0])
        return (score, offset), subpipe
