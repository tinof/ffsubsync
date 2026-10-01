"""Tests for the Claude judge of ssync's AI mode. Claude is never called."""

import json
import subprocess

import pytest

from ffsubsync.ai_judge import (
    JudgeCue,
    JudgeUnavailable,
    anchors_from_matches,
    build_prompts,
    judge_command,
    run_claude,
)

SUB = [
    JudgeCue(10.0, 12.0, "Tarkista pulssi."),
    JudgeCue(13.0, 15.0, "Kiitos, pojat.\nToisen pitäisi selvitä."),
    JudgeCue(16.0, 18.0, "TURVALLISUUSOPAS"),
    JudgeCue(300.0, 302.0, "Tässä on kaikki."),
]
REF = [
    JudgeCue(60.0, 62.0, "Check pulse."),
    JudgeCue(63.2, 65.0, "Thanks, lads."),
    JudgeCue(200.0, 202.0, "Where is the dog?"),
    JudgeCue(350.0, 352.0, "It's all there, yeah."),
]


def _completed(stdout, returncode=0, stderr=""):
    return subprocess.CompletedProcess(["claude"], returncode, stdout, stderr)


def _runner(result, calls=None):
    def runner(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    return runner


def _found(_name):
    return "/usr/bin/claude"


class TestBuildPrompts:
    def test_window_lists_its_cues_and_the_reference_around_them(self):
        prompts = build_prompts([(0, 2)], SUB, REF, [50.0, 50.0, 50.0, 50.0])

        assert len(prompts) == 1
        prompt = prompts[0]
        assert "S1 [01:00.0] Tarkista pulssi." in prompt
        assert "S2 [01:03.0] Kiitos, pojat. / Toisen pitäisi selvitä." in prompt
        assert "R1 [01:00.0] Check pulse." in prompt
        assert "R2 [01:03.2] Thanks, lads." in prompt
        # Outside the window and its 90 s margin.
        assert "S4 " not in prompt
        assert "R4 " not in prompt

    def test_windows_are_split_over_prompts_by_size(self):
        prompts = build_prompts(
            [(0, 1), (2, 3)], SUB, REF, [50.0] * 4, max_cues_per_prompt=2
        )

        assert len(prompts) == 2
        assert "S1 " in prompts[0] and "S3 " not in prompts[0]
        assert "S3 " in prompts[1] and "S4 " in prompts[1]

    def test_no_windows_no_prompts(self):
        assert build_prompts([], SUB, REF, [0.0] * 4) == []


class TestJudgeCommand:
    def test_headless_without_tools_or_settings(self):
        cmd = judge_command("/usr/bin/claude", model=None, budget_usd=1.0)

        assert cmd[:2] == ["/usr/bin/claude", "-p"]
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert (
            "matches" in json.loads(cmd[cmd.index("--json-schema") + 1])["properties"]
        )
        assert cmd[cmd.index("--max-budget-usd") + 1] == "1"
        assert "--strict-mcp-config" in cmd
        assert "--no-session-persistence" in cmd
        assert "--model" not in cmd

    def test_model_is_passed_when_given(self):
        cmd = judge_command("claude", model="opus", budget_usd=None)

        assert cmd[cmd.index("--model") + 1] == "opus"
        assert "--max-budget-usd" not in cmd


class TestRunClaude:
    def test_returns_matches_from_structured_output(self):
        calls = []
        envelope = {
            "is_error": False,
            "structured_output": {"matches": [{"sub": 1, "ref": 1}]},
            "total_cost_usd": 0.01,
        }

        matches = run_claude(
            "prompt text",
            runner=_runner(_completed(json.dumps(envelope)), calls),
            which=_found,
            cwd="/tmp",
            timeout=12,
        )

        assert matches == [{"sub": 1, "ref": 1}]
        cmd, kwargs = calls[0]
        assert cmd[0] == "/usr/bin/claude"
        assert kwargs["input"] == "prompt text"
        assert kwargs["timeout"] == 12
        assert kwargs["cwd"] == "/tmp"

    def test_falls_back_to_json_in_result_text(self):
        envelope = {"result": json.dumps({"matches": [{"sub": 2, "ref": None}]})}

        matches = run_claude(
            "p", runner=_runner(_completed(json.dumps(envelope))), which=_found
        )

        assert matches == [{"sub": 2, "ref": None}]

    def test_missing_cli(self):
        with pytest.raises(JudgeUnavailable, match="not on PATH"):
            run_claude("p", which=lambda _name: None)

    @pytest.mark.parametrize(
        "result, message",
        [
            (subprocess.TimeoutExpired("claude", 5), "timed out"),
            (OSError("boom"), "could not be started"),
            (_completed("", returncode=1, stderr="not logged in"), "not logged in"),
            (_completed("this is not json"), "did not return JSON"),
            (_completed(json.dumps({"is_error": True, "result": "limit"})), "limit"),
            (_completed(json.dumps({"result": "plain words"})), "no structured"),
            (_completed(json.dumps({"structured_output": {"x": 1}})), "no 'matches'"),
        ],
    )
    def test_failures_raise_judge_unavailable(self, result, message):
        with pytest.raises(JudgeUnavailable, match=message):
            run_claude("p", runner=_runner(result), which=_found, timeout=5)


class TestAnchorsFromMatches:
    def _anchors(self, matches, offsets=(50.0, 50.0, 50.0, 50.0), windows=((0, 3),)):
        return anchors_from_matches(
            matches,
            list(windows),
            SUB,
            REF,
            list(offsets),
            min_offset=-60.0,
            max_offset=120.0,
        )

    def test_offset_is_reference_start_minus_cue_start(self):
        anchors = self._anchors([{"sub": 1, "ref": 1}, {"sub": 2, "ref": 2}])

        assert anchors == pytest.approx({0: 50.0, 1: 50.2})

    def test_null_and_malformed_matches_are_ignored(self):
        anchors = self._anchors(
            [
                {"sub": 1, "ref": None},
                {"sub": "2", "ref": 2},
                {"sub": 2, "ref": 99},
                {"ref": 1},
                {"sub": True, "ref": 1},
            ]
        )

        assert anchors == {}

    def test_cue_outside_the_asked_windows_is_ignored(self):
        anchors = self._anchors([{"sub": 4, "ref": 4}], windows=((0, 1),))

        assert anchors == {}

    def test_offset_outside_the_search_range_is_dropped(self):
        # S1 -> R3 would be +190 s, beyond max_offset.
        assert self._anchors([{"sub": 1, "ref": 3}]) == {}

    def test_lone_match_that_contradicts_the_aligner_is_dropped(self):
        # S1 -> R1 is +50 s but the aligner has the cue at 0 s and no neighbour
        # match backs it up.
        assert self._anchors([{"sub": 1, "ref": 1}], offsets=(0.0,) * 4) == {}

    def test_neighbouring_matches_back_each_other_up(self):
        anchors = self._anchors(
            [{"sub": 1, "ref": 1}, {"sub": 2, "ref": 2}], offsets=(0.0,) * 4
        )

        assert sorted(anchors) == [0, 1]
