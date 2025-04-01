# -*- coding: utf-8 -*-
from datetime import timedelta
import logging
import numbers

from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin
import bisect
from datetime import timedelta
import logging
import numbers
from typing import List, Tuple

import numpy as np

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin
from ffsubsync.sklearn_shim import TransformerMixin

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


# Helper function to convert timedelta to samples
def _td_to_samples(td: timedelta, sample_rate: int) -> int:
    return int(td.total_seconds() * sample_rate)

# Helper function to convert samples to timedelta
def _samples_to_td(samples: float, sample_rate: int) -> timedelta:
    return timedelta(seconds=samples / sample_rate)


class SubtitleShifter(SubsMixin, TransformerMixin):
    def __init__(self, td_seconds):
        super(SubsMixin, self).__init__()
        if not isinstance(td_seconds, timedelta):
            self.td_seconds = timedelta(seconds=td_seconds)
        else:
            self.td_seconds = td_seconds

    def fit(self, subs: GenericSubtitlesFile, *_):
        self.subs_ = subs.offset(self.td_seconds)
        return self

    def transform(self, *_):
        return self.subs_


class SubtitleWarper(SubsMixin, TransformerMixin):
    """
    Applies non-linear time warping to subtitles based on a timing map.
    The timing map consists of anchor points mapping reference time to subtitle time.
    """
    def __init__(self, timing_map: List[Tuple[int, int]], sample_rate: int = SAMPLE_RATE):
        """
        Args:
            timing_map: List of (reference_sample, subtitle_sample) anchor points,
                        sorted by reference_sample.
            sample_rate: Sample rate used for VAD signals.
        """
        super(SubsMixin, self).__init__()
        # Ensure map is sorted by subtitle samples for efficient lookup
        self.timing_map_ = sorted(timing_map, key=lambda x: x[1])
        self.sample_rate = sample_rate
        # Extract subtitle sample times for bisect
        self.sub_samples_ = [anchor[1] for anchor in self.timing_map_]
        if not self.timing_map_:
             raise ValueError("Timing map cannot be empty for SubtitleWarper")
        logger.info(f"SubtitleWarper initialized with {len(self.timing_map_)} anchors.")

    def _warp_time_samples(self, original_sub_samples: int) -> int:
        """
        Maps an original subtitle time (in samples) to the warped reference time (in samples)
        using linear interpolation between anchor points.
        """
        # Find the segment the original time falls into using binary search
        # Find insertion point for original_sub_samples in the sorted list of subtitle anchor times
        # `bisect_left` finds the index `i` such that all `a[j]` with `j < i` have `a[j] < x`,
        # and all `a[k]` with `k >= i` have `a[k] >= x`.
        idx = bisect.bisect_left(self.sub_samples_, original_sub_samples)

        # Handle edge cases: time is before the first anchor or at/after the last anchor
        if idx == 0:
            # Time is before or exactly at the first anchor point
            return self.timing_map_[0][0]
        if idx >= len(self.timing_map_):
            # Time is at or after the last anchor point
            return self.timing_map_[-1][0]

        # Interpolate between anchor points timing_map_[idx-1] and timing_map_[idx]
        ref1, sub1 = self.timing_map_[idx - 1]
        ref2, sub2 = self.timing_map_[idx]

        # Avoid division by zero if anchor points have the same subtitle time (shouldn't happen with distinct anchors)
        if sub2 == sub1:
            return ref1

        # Calculate interpolation ratio based on subtitle time
        ratio = (original_sub_samples - sub1) / (sub2 - sub1)

        # Calculate the warped reference time
        warped_ref_samples = ref1 + ratio * (ref2 - ref1)

        return int(round(warped_ref_samples))

    def fit(self, subs: GenericSubtitlesFile, *_):
        """
        Applies the time warping to the provided subtitles.
        """
        warped_subs = []
        logger.info(f"Applying time warping to {len(subs)} subtitles...")
        for i, sub in enumerate(subs):
            original_start_samples = _td_to_samples(sub.start, self.sample_rate)
            original_end_samples = _td_to_samples(sub.end, self.sample_rate)

            new_start_samples = self._warp_time_samples(original_start_samples)
            new_end_samples = self._warp_time_samples(original_end_samples)

            # Ensure start time is not after end time after warping
            if new_end_samples < new_start_samples:
                # This can happen if warping is extreme or anchors are very close
                # Option 1: Set duration to zero at the start time
                # Option 2: Keep original duration?
                # Option 3: Use the midpoint?
                # Let's try setting duration to a minimal value (e.g., 1 sample) for now
                logger.warning(f"Subtitle {i+1}: End time warped before start time ({new_end_samples} < {new_start_samples}). Clamping end time.")
                new_end_samples = new_start_samples + 1 # Minimal duration

            new_start_td = _samples_to_td(new_start_samples, self.sample_rate)
            new_end_td = _samples_to_td(new_end_samples, self.sample_rate)

            warped_subs.append(
                GenericSubtitle(
                    start=new_start_td,
                    end=new_end_td,
                    inner=sub.inner,
                )
            )
            if i < 5 or i % 100 == 0: # Log first few and periodically
                 logger.debug(f"Sub {i+1}: Orig ({sub.start.total_seconds():.3f} -> {sub.end.total_seconds():.3f}) Warped ({new_start_td.total_seconds():.3f} -> {new_end_td.total_seconds():.3f})")


        self.subs_ = subs.clone_props_for_subs(warped_subs)
        logger.info("Time warping complete.")
        return self

    def transform(self, *_):
        return self.subs_


class SubtitleScaler(SubsMixin, TransformerMixin):
    def __init__(self, scale_factor):
        assert isinstance(scale_factor, numbers.Number)
        super(SubsMixin, self).__init__()
        self.scale_factor = scale_factor

    def fit(self, subs: GenericSubtitlesFile, *_):
        scaled_subs = []
        for sub in subs:
            scaled_subs.append(
                GenericSubtitle(
                    # py2 doesn't support direct multiplication of timedelta w/ float
                    timedelta(seconds=sub.start.total_seconds() * self.scale_factor),
                    timedelta(seconds=sub.end.total_seconds() * self.scale_factor),
                    sub.inner,
                )
            )
        self.subs_ = subs.clone_props_for_subs(scaled_subs)
        return self

    def transform(self, *_):
        return self.subs_


class SubtitleMerger(SubsMixin, TransformerMixin):
    def __init__(self, reference_subs, first="reference"):
        assert first in ("reference", "output")
        super(SubsMixin, self).__init__()
        self.reference_subs = reference_subs
        self.first = first

    def fit(self, output_subs: GenericSubtitlesFile, *_):
        def _merger_gen(a, b):
            ita, itb = iter(a), iter(b)
            cur_a = next(ita, None)
            cur_b = next(itb, None)
            while True:
                if cur_a is None and cur_b is None:
                    return
                elif cur_a is None:
                    while cur_b is not None:
                        yield cur_b
                        cur_b = next(itb, None)
                    return
                elif cur_b is None:
                    while cur_a is not None:
                        yield cur_a
                        cur_a = next(ita, None)
                    return
                # else: neither are None
                if cur_a.start < cur_b.start:
                    swapped = False
                else:
                    swapped = True
                    cur_a, cur_b = cur_b, cur_a
                    ita, itb = itb, ita
                prev_a = cur_a
                while prev_a is not None and cur_a.start < cur_b.start:
                    cur_a = next(ita, None)
                    if cur_a is None or cur_a.start < cur_b.start:
                        yield prev_a
                        prev_a = cur_a
                if prev_a is None:
                    while cur_b is not None:
                        yield cur_b
                        cur_b = next(itb, None)
                    return
                if cur_b.start - prev_a.start < cur_a.start - cur_b.start:
                    if swapped:
                        yield cur_b.merge_with(prev_a)
                        ita, itb = itb, ita
                        cur_a, cur_b = cur_b, cur_a
                        cur_a = next(ita, None)
                    else:
                        yield prev_a.merge_with(cur_b)
                        cur_b = next(itb, None)
                else:
                    if swapped:
                        yield cur_b.merge_with(cur_a)
                        ita, itb = itb, ita
                    else:
                        yield cur_a.merge_with(cur_b)
                    cur_a = next(ita, None)
                    cur_b = next(itb, None)

        merged_subs = []
        if self.first == "reference":
            first, second = self.reference_subs, output_subs
        else:
            first, second = output_subs, self.reference_subs
        for merged in _merger_gen(first, second):
            merged_subs.append(merged)
        self.subs_ = output_subs.clone_props_for_subs(merged_subs)
        return self

    def transform(self, *_):
        return self.subs_
