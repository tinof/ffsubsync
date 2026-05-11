import logging
import math
from collections import defaultdict

import numpy as np

from ffsubsync.constants import FRAMERATE_RATIOS, FRAMERATE_SNAP_TOLERANCE
from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1
OFF_GRID_GSS_MIN_IMPROVEMENT = 1.15

# All known valid framerate ratios and their reciprocals, plus 1.0.
_KNOWN_FRAMERATE_RATIOS: list[float] = (
    [1.0] + list(FRAMERATE_RATIOS) + [1.0 / r for r in FRAMERATE_RATIOS]
)


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: int | None = None) -> None:
        self.max_offset_samples: int | None = max_offset_samples
        self.best_offset_: int | None = None
        self.best_score_: float | None = None
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

    def transform(self, *_) -> int | tuple[float, int]:
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

    A result is only accepted when a strict majority of windows (> 50%)
    agree on the offset bin. If no majority is reached, raises
    FailedToFindAlignmentException so the caller can fall back to a
    simpler strategy.
    """

    def __init__(
        self,
        max_offset_samples: int | None = None,
        window_size_seconds: int = 600,
        overlap_seconds: int = 300,
        sample_rate: int = 100,
        tolerance_seconds: float = 0.5,
    ) -> None:
        self.max_offset_samples: int | None = max_offset_samples
        self.window_size_seconds: int = window_size_seconds
        self.overlap_seconds: int = overlap_seconds
        self.sample_rate: int = sample_rate
        self.tolerance_seconds: float = tolerance_seconds
        self.best_offset_: int | None = None
        self.best_score_: float | None = None
        self.get_score_: bool = False
        self.confidence_: str = "unknown"
        self.vote_ratio_: float = 0.0

    def fit(self, refstring, substring, get_score: bool = False) -> "SegmentedAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = (
            2 * np.array(s).astype(float) - 1 for s in [refstring, substring]
        )

        window_size_samples = int(self.window_size_seconds * self.sample_rate)
        step_size_samples = int(
            (self.window_size_seconds - self.overlap_seconds) * self.sample_rate
        )
        tolerance_samples = int(self.tolerance_seconds * self.sample_rate)

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
            self.confidence_ = "high"
            self.vote_ratio_ = 1.0
            self.get_score_ = get_score
            return self

        if step_size_samples <= 0:
            raise ValueError("segment overlap must be smaller than segment window")

        window_results: list[tuple[int, float]] = []

        logger.info(
            "Running segmented alignment: window_size=%ds, overlap=%ds, tolerance=%.1fs",
            self.window_size_seconds,
            self.overlap_seconds,
            self.tolerance_seconds,
        )

        num_windows = 0
        last_start_idx = len(refstring) - window_size_samples
        window_starts = list(range(0, last_start_idx + 1, step_size_samples))
        if window_starts[-1] != last_start_idx:
            window_starts.append(last_start_idx)

        for start_idx in window_starts:
            end_idx = min(start_idx + window_size_samples, len(refstring))

            ref_window = refstring[start_idx:end_idx]
            sub_window = (
                substring[start_idx:end_idx]
                if end_idx <= len(substring)
                else substring[start_idx:]
            )

            if len(sub_window) < window_size_samples // 2:
                continue

            try:
                aligner = FFTAligner(max_offset_samples=self.max_offset_samples)
                aligner.fit(ref_window, sub_window, get_score=True)

                if aligner.best_offset_ is not None and aligner.best_score_ is not None:
                    window_results.append((aligner.best_offset_, aligner.best_score_))
                    num_windows += 1

            except Exception as e:
                logger.warning("Failed to align window %d: %s", num_windows, e)
                continue

        if len(window_results) == 0:
            self.confidence_ = "low"
            raise FailedToFindAlignmentException(
                "Segmented alignment failed: no windows could be aligned"
            )

        logger.info("Aligned %d windows, aggregating results...", num_windows)

        offset_bins: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)

        for offset, score in window_results:
            bin_center = round(offset / tolerance_samples) * tolerance_samples
            offset_bins[bin_center].append((offset, score))
            logger.debug(
                "Window offset=%d (bin=%d), score=%.0f", offset, bin_center, score
            )

        bin_votes: dict[int, int] = {
            bin_center: len(results) for bin_center, results in offset_bins.items()
        }

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

        # Confidence gate: require strict majority (> 50% of windows must agree).
        min_required_votes = (num_windows // 2) + 1
        if max_votes < min_required_votes:
            self.confidence_ = "low"
            self.vote_ratio_ = max_votes / num_windows
            raise FailedToFindAlignmentException(
                f"Segmented alignment has low confidence: {max_votes}/{num_windows} windows "
                f"agreed (need >= {min_required_votes}). Windows proposed widely different offsets."
            )

        self.vote_ratio_ = max_votes / num_windows
        if max_votes == num_windows:
            self.confidence_ = "high"
        else:
            self.confidence_ = "medium"

        if len(winning_bins) > 1:
            bin_scores = {
                bin_center: sum(score for _, score in offset_bins[bin_center])
                for bin_center in winning_bins
            }
            winning_bin = max(bin_scores, key=lambda b: bin_scores[b])
            logger.info(
                "Tie broken by score: bin %d selected with cumulative score %.0f",
                winning_bin,
                bin_scores[winning_bin],
            )
        else:
            winning_bin = winning_bins[0]

        winning_results = offset_bins[winning_bin]
        self.best_offset_ = int(np.mean([offset for offset, _ in winning_results]))
        mean_score = float(np.mean([score for _, score in winning_results]))
        self.best_score_ = mean_score * self.vote_ratio_

        logger.info(
            "Consensus: offset=%d samples (%.2fs), score=%.0f "
            "(mean=%.0f, vote_ratio=%.2f) from %d/%d windows [confidence=%s]",
            self.best_offset_,
            self.best_offset_ / self.sample_rate,
            self.best_score_,
            mean_score,
            self.vote_ratio_,
            len(winning_results),
            num_windows,
            self.confidence_,
        )

        self.get_score_ = get_score
        return self

    def transform(self, *_) -> int | tuple[float, int]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(
        self,
        base_aligner: FFTAligner | type[FFTAligner],
        srtin: str | None = None,
        sample_rate=None,
        max_offset_seconds=None,
        **aligner_kwargs,
    ) -> None:
        self.srtin: str | None = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: int | None = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(
                max_offset_samples=self.max_offset_samples, **aligner_kwargs
            )
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: int | None = max_offset_seconds
        self._scores: list[tuple[tuple[float, int], Pipeline]] = []

    def fit_gss(self, refstring, subpipe_maker):
        # Track (score, subpipe, ratio) for post-GSS plausibility check.
        gss_candidates: list[tuple[tuple[float, int], Pipeline, float]] = []
        best_baseline_score = max((score[0] for score, _ in self._scores), default=None)

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
                gss_candidates.append((score, subpipe, framerate_ratio))
            return -score[0]

        gss(opt_func, MIN_FRAMERATE_RATIO, MAX_FRAMERATE_RATIO)

        # Discard ratios that are not near any known framerate pair.
        for score, subpipe, ratio in gss_candidates:
            nearest = min(_KNOWN_FRAMERATE_RATIOS, key=lambda c: abs(ratio - c) / c)
            rel_err = abs(ratio - nearest) / nearest
            if rel_err > FRAMERATE_SNAP_TOLERANCE:
                if (
                    best_baseline_score is not None
                    and score[0] >= best_baseline_score * OFF_GRID_GSS_MIN_IMPROVEMENT
                ):
                    logger.warning(
                        "GSS ratio %.4f not near any known framerate pair "
                        "(closest=%.4f, err=%.2f%%), but accepting because "
                        "score %.0f beats best fixed-ratio score %.0f by >= %.0f%%",
                        ratio,
                        nearest,
                        rel_err * 100,
                        score[0],
                        best_baseline_score,
                        (OFF_GRID_GSS_MIN_IMPROVEMENT - 1.0) * 100,
                    )
                    self._scores.append((score, subpipe))
                else:
                    logger.warning(
                        "GSS ratio %.4f not near any known framerate pair "
                        "(closest=%.4f, err=%.2f%%); discarding to avoid drift",
                        ratio,
                        nearest,
                        rel_err * 100,
                    )
            else:
                if abs(ratio - nearest) > 1e-6:
                    logger.info(
                        "GSS ratio %.4f snapped to known pair %.4f (err=%.3f%%)",
                        ratio,
                        nearest,
                        rel_err * 100,
                    )
                    subpipe = subpipe_maker(nearest)
                    substring = subpipe.fit_transform(self.srtin)
                    score = self.base_aligner.fit_transform(
                        refstring, substring, get_score=True
                    )
                    logger.info(
                        "got score %.0f (offset %d) for snapped ratio %.4f",
                        score[0],
                        score[1],
                        nearest,
                    )
                self._scores.append((score, subpipe))

        return self

    def fit(self, refstring, subpipes: Pipeline | list[Pipeline]) -> "MaxScoreAligner":
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
            try:
                score = self.base_aligner.fit_transform(
                    refstring, substring, get_score=True
                )
                self._scores.append((score, subpipe))
            except FailedToFindAlignmentException as e:
                logger.debug("Skipping subpipe: low-confidence alignment (%s)", e)
                continue
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
