"""Tests for the cut-aware aligner (subtitle of one cut vs a text track of another)."""

import json
from itertools import pairwise
from pathlib import Path

import pytest

from ffsubsync.cut_aligner import (
    assess_cut_alignment,
    compute_cut_offsets,
    default_offset_range,
    is_dialogue_text,
    review_windows,
)

FIXTURE = Path(__file__).parent / "data" / "cut_blue_lights_s01e02.json"


def _scene(start, n_cues, gap=3.2, duration=2.4):
    return [(start + i * gap, start + i * gap + duration) for i in range(n_cues)]


def _longer_cut():
    """A subtitle of three scenes, and a reference with a scene added twice.

    Returns (sub_times, ref_times, true offset per cue).
    """
    scenes = [_scene(10.0, 20), _scene(80.0, 25, gap=2.9), _scene(160.0, 20, gap=3.5)]
    added = [0.0, 45.0, 45.0 + 70.0]
    sub, ref, truth = [], [], []
    for scene, shift in zip(scenes, added, strict=True):
        sub.extend(scene)
        ref.extend((a + shift, b + shift) for a, b in scene)
        truth.extend([shift] * len(scene))
    # The scenes only the longer cut has.
    ref.extend(_scene(78.0, 12, gap=3.7))
    ref.extend(_scene(210.0, 15, gap=4.1))
    return sub, sorted(ref), truth


class TestIsDialogueText:
    @pytest.mark.parametrize(
        "text",
        ["Thanks, lads.", "OK?", "27, 28!", "I...", "RADIO:  Bravo Lima 76."],
    )
    def test_spoken_lines_are_dialogue(self, text):
        assert is_dialogue_text(text)

    @pytest.mark.parametrize(
        "text", ["SIREN WAILS", "SHE EXHALES SMOKE", "[music]", "(door slams)", "  "]
    )
    def test_sound_descriptions_are_not(self, text):
        assert not is_dialogue_text(text)


class TestComputeCutOffsets:
    def test_recovers_forward_jumps_at_added_scenes(self):
        sub, ref, truth = _longer_cut()

        alignment = compute_cut_offsets(sub, ref)

        assert alignment.offsets == pytest.approx(truth, abs=0.05)
        assert [s.offset_seconds for s in alignment.segments] == pytest.approx(
            [0.0, 45.0, 115.0], abs=0.05
        )
        assert [(s.first, s.last) for s in alignment.segments] == [
            (0, 19),
            (20, 44),
            (45, 64),
        ]

    def test_same_cut_with_one_offset_is_a_single_segment(self):
        sub = _scene(10.0, 40)
        ref = [(a + 3.2 * 0 + 7.0, b + 7.0) for a, b in sub]

        alignment = compute_cut_offsets(sub, ref)

        assert len(alignment.segments) == 1
        assert alignment.segments[0].offset_seconds == pytest.approx(7.0, abs=0.05)
        assert alignment.segments[0].agreement == 1.0

    def test_no_cues(self):
        alignment = compute_cut_offsets([], [(1.0, 2.0)])

        assert alignment.offsets == []
        assert alignment.segments == []

    def test_anchor_moves_a_cue_to_the_named_scene(self):
        # Two identical runs of cues in the reference: timing cannot tell which
        # one the subtitle belongs to, an anchor can.
        sub = _scene(10.0, 8)
        ref = sorted(_scene(10.0, 8) + _scene(100.0, 8))

        free = compute_cut_offsets(sub, ref)
        anchored = compute_cut_offsets(sub, ref, anchors={3: 90.0})

        assert free.offsets == pytest.approx([0.0] * 8, abs=0.05)
        assert anchored.offsets[3] == pytest.approx(90.0, abs=0.05)

    def test_small_step_back_is_allowed_large_is_not(self):
        sub = _scene(10.0, 10) + _scene(60.0, 30)
        # First scene 0.6 s later than the rest.
        ref = [(a + 5.6, b + 5.6) for a, b in sub[:10]] + [
            (a + 5.0, b + 5.0) for a, b in sub[10:]
        ]

        alignment = compute_cut_offsets(sub, ref)
        forward_only = compute_cut_offsets(sub, ref, max_back_step=0.0)

        assert alignment.offsets[0] == pytest.approx(5.6, abs=0.05)
        assert alignment.offsets[-1] == pytest.approx(5.0, abs=0.05)
        assert all(
            b >= a
            for a, b in zip(
                forward_only.offsets, forward_only.offsets[1:], strict=False
            )
        )

    def test_default_range_covers_the_length_difference(self):
        sub, ref, _ = _longer_cut()

        lo, hi = default_offset_range(sub, ref)

        assert lo == -60.0
        assert hi == pytest.approx(115.0 + 60.0)


class TestAssessAndReview:
    def test_assessment_reports_agreement_and_uncovered_scenes(self):
        sub, ref, truth = _longer_cut()

        assessment = assess_cut_alignment(sub, ref, truth)

        assert assessment.within_half_second == 1.0
        assert assessment.within_one_second == 1.0
        # The two added scenes have no subtitle.
        assert [count for _, _, count in assessment.uncovered] == [12, 15]

    def test_unshifted_subtitle_scores_low(self):
        sub, ref, _ = _longer_cut()

        assessment = assess_cut_alignment(sub, ref, [20.3] * len(sub))

        assert assessment.within_one_second < 0.7

    def test_review_windows_surround_jumps_and_the_opening(self):
        sub, ref, _ = _longer_cut()
        alignment = compute_cut_offsets(sub, ref)

        windows = review_windows(alignment)

        assert windows == [(0, 9), (16, 23), (41, 48)]

    def test_single_segment_needs_no_review(self):
        sub = _scene(10.0, 40)
        alignment = compute_cut_offsets(sub, [(a + 7.0, b + 7.0) for a, b in sub])

        assert review_windows(alignment) == []


class TestBlueLightsRegression:
    """Timings of a real episode: a Finnish subtitle of the broadcast cut against
    the English track of a cut that is 5.5 minutes longer (22 offsets, -5 s to
    +329 s). The expected offsets were verified by hand against text and audio.
    """

    @pytest.fixture(scope="class")
    def data(self):
        raw = json.loads(FIXTURE.read_text())
        return {
            "sub": [(a, b) for a, b in raw["sub"]],
            "ref": [(a, b) for a, b, is_dialogue in raw["ref"] if is_dialogue],
            "expected": raw["expected_offsets"],
        }

    # Cues 445-459: one conversation with three short trims. The timing score
    # alone settles on a neighbouring offset there; the judge's anchors fix it.
    JUDGE_ONLY = range(444, 459)

    def test_aligner_alone_matches_outside_the_judged_stretch(self, data):
        alignment = compute_cut_offsets(data["sub"], data["ref"])

        wrong = [
            i
            for i, (got, want) in enumerate(
                zip(alignment.offsets, data["expected"], strict=True)
            )
            if abs(got - want) > 0.5 and i not in self.JUDGE_ONLY
        ]
        assert wrong == []
        assert alignment.offsets[0] == pytest.approx(-4.6, abs=0.5)
        assert alignment.offsets[-1] == pytest.approx(329.24, abs=0.1)

    def test_anchors_fix_the_judged_stretch(self, data):
        anchors = {i: data["expected"][i] for i in (444, 447, 450, 453, 455, 456, 458)}

        alignment = compute_cut_offsets(data["sub"], data["ref"], anchors=anchors)

        errors = [
            abs(got - want)
            for got, want in zip(alignment.offsets, data["expected"], strict=True)
        ]
        assert max(errors) <= 1.0
        assert sum(e > 0.5 for e in errors) <= 3

    def test_the_judged_stretch_is_offered_for_review(self, data):
        alignment = compute_cut_offsets(data["sub"], data["ref"])

        reviewed = {
            i
            for first, last in review_windows(alignment)
            for i in range(first, last + 1)
        }

        assert {444, 445, 446} <= reviewed
        assert len(reviewed) < len(data["sub"]) / 2

    def test_score_per_cue_separates_a_matching_subtitle_from_chance(self, data):
        import random

        sub = data["sub"]
        gaps = [b[0] - a[0] for a, b in pairwise(sub)]
        random.Random(1).shuffle(gaps)
        shuffled, t = [], 5.0
        for (start, end), gap in zip(sub, [*gaps, 3.0], strict=True):
            shuffled.append((t, t + end - start))
            t += gap

        matching = compute_cut_offsets(sub, data["ref"])
        chance = compute_cut_offsets(shuffled, data["ref"])

        assert matching.score_per_cue > 1.3
        assert chance.score_per_cue < 0.9

    def test_assessment_passes_the_gate(self, data):
        assessment = assess_cut_alignment(data["sub"], data["ref"], data["expected"])

        assert assessment.within_one_second > 0.85
        assert len(assessment.uncovered) >= 3
