import logging
import numbers
from datetime import timedelta

import numpy as np

from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin
from ffsubsync.sklearn_shim import TransformerMixin

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


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


class PiecewiseSubtitleShifter(SubsMixin, TransformerMixin):
    """Shift cues by a time-varying offset interpolated between anchors.

    ``anchors`` is a sequence of ``(time_seconds, offset_seconds)`` knots in
    reference time. Offsets are linearly interpolated over cue start times and
    held constant beyond the first and last anchor. Cue durations are
    preserved exactly.
    """

    def __init__(self, anchors):
        super(SubsMixin, self).__init__()
        anchors = sorted(anchors, key=lambda a: a[0])
        if len(anchors) < 2:
            raise ValueError("piecewise shifting needs at least two anchors")
        self.anchor_times = [float(t) for t, _ in anchors]
        self.anchor_offsets = [float(o) for _, o in anchors]

    def offset_at(self, seconds: float) -> float:
        return float(np.interp(seconds, self.anchor_times, self.anchor_offsets))

    def fit(self, subs: GenericSubtitlesFile, *_):
        shifted_subs = []
        for sub in subs:
            start = sub.start.total_seconds()
            duration = sub.end.total_seconds() - start
            new_start = max(0.0, start + self.offset_at(start))
            shifted_subs.append(
                GenericSubtitle(
                    timedelta(seconds=new_start),
                    timedelta(seconds=new_start + duration),
                    sub.inner,
                )
            )
        self.subs_ = subs.clone_props_for_subs(shifted_subs)
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
