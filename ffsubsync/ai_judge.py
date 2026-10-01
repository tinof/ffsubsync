"""Claude as a judge for the cut-aware aligner.

:mod:`ffsubsync.cut_aligner` places cues by timing alone. Around a jump, in a
short segment or in a recap, the timing score can prefer the wrong scene. A
reader who understands both languages sees it at once, so the doubtful windows
go to Claude through the local ``claude`` CLI in headless mode (``claude -p``).

Claude gets no tools and returns no numbers. It sees the cues of a window
(``S<n>``) next to the reference cues around them (``R<n>``) and answers which
``R`` cue each ``S`` cue translates. The offsets follow from the cue timings
here, and they return to the aligner as anchors, so a wrong answer can only move
a cue onto a real reference cue start.

Only the text of the reviewed windows leaves the machine.
"""

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

CLAUDE_BINARY = "claude"
DEFAULT_TIMEOUT_SECONDS: float = 300.0
DEFAULT_BUDGET_USD: float = 1.0
# Reference cues this far before and after a window are shown with it.
REFERENCE_MARGIN_SECONDS: float = 90.0
# Windows are packed into prompts of at most this many subtitle cues.
MAX_CUES_PER_PROMPT: int = 200
# An anchor must agree this closely with the aligner or with a neighbour anchor.
ANCHOR_CONSENSUS_SECONDS: float = 1.0

SYSTEM_PROMPT = (
    "You match subtitle lines across languages. You answer only with the "
    "requested JSON."
)

INSTRUCTIONS = """\
Two subtitle tracks of the same programme follow, in different languages. The S
cues were made for a different cut, so scenes may be missing on either side and
the S times are only roughly right.

For every S cue, find the R cue that begins with the same line: the R cue whose
first words say what the first words of the S cue say. Use meaning, names and
numbers, not the times.

Answer null for an S cue when
- no R cue says it (on-screen text, credits, a line the other track leaves out),
- it begins in the middle of an R cue rather than at its start, or
- you are not sure.

A wrong match does harm and null does none. Give one entry for every S cue.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub": {"type": "integer"},
                    "ref": {"type": ["integer", "null"]},
                },
                "required": ["sub", "ref"],
            },
        }
    },
    "required": ["matches"],
}


class JudgeUnavailable(Exception):
    """The judge could not be asked or gave no usable answer."""


@dataclass(frozen=True)
class JudgeCue:
    start: float
    end: float
    text: str


Runner = Callable[..., "subprocess.CompletedProcess[str]"]
# What ssync injects: prompt in, [{"sub": int, "ref": int | None}, ...] out.
Judge = Callable[[str], Sequence[Mapping[str, Any]]]


def _clock(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    return f"{int(seconds) // 60:02d}:{seconds % 60:04.1f}"


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())


def build_prompts(
    windows: Sequence[tuple[int, int]],
    sub_cues: Sequence[JudgeCue],
    ref_cues: Sequence[JudgeCue],
    offsets: Sequence[float],
    *,
    max_cues_per_prompt: int = MAX_CUES_PER_PROMPT,
    margin_seconds: float = REFERENCE_MARGIN_SECONDS,
) -> list[str]:
    """Prompts for the windows (0-based inclusive cue ranges), numbered from 1.

    S and R numbers in the prompt are the 1-based positions in ``sub_cues`` and
    ``ref_cues``. S times are shown with the aligner's ``offsets`` applied, so
    matching lines are near each other.
    """
    batches: list[list[tuple[int, int]]] = []
    count = 0
    for first, last in windows:
        size = last - first + 1
        if not batches or count + size > max_cues_per_prompt:
            batches.append([])
            count = 0
        batches[-1].append((first, last))
        count += size

    prompts = []
    for batch in batches:
        parts = [INSTRUCTIONS]
        for number, (first, last) in enumerate(batch, start=1):
            shifted = [
                (sub_cues[i].start + offsets[i], sub_cues[i].end + offsets[i])
                for i in range(first, last + 1)
            ]
            lo = min(s for s, _ in shifted) - margin_seconds
            hi = max(e for _, e in shifted) + margin_seconds
            parts.append(f"## Window {number}\n")
            parts.append("S cues:")
            for i, (start, _) in zip(range(first, last + 1), shifted, strict=True):
                parts.append(
                    f"S{i + 1} [{_clock(start)}] {_one_line(sub_cues[i].text)}"
                )
            parts.append("\nR cues:")
            for j, cue in enumerate(ref_cues):
                if lo <= cue.start <= hi:
                    parts.append(
                        f"R{j + 1} [{_clock(cue.start)}] {_one_line(cue.text)}"
                    )
            parts.append("")
        prompts.append("\n".join(parts))
    return prompts


def judge_command(
    binary: str, *, model: str | None, budget_usd: float | None
) -> list[str]:
    """The headless Claude call: no tools, no settings, schema-checked JSON out."""
    cmd = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(RESPONSE_SCHEMA),
        "--system-prompt",
        SYSTEM_PROMPT,
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if budget_usd is not None:
        cmd += ["--max-budget-usd", f"{budget_usd:g}"]
    if model:
        cmd += ["--model", model]
    return cmd


def run_claude(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    budget_usd: float | None = DEFAULT_BUDGET_USD,
    cwd: str | os.PathLike[str] | None = None,
    runner: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Mapping[str, Any]]:
    """Ask Claude one prompt and return its ``matches`` list.

    Raises:
        JudgeUnavailable: The CLI is missing, failed, timed out, or answered
            with something other than the requested JSON.
    """
    binary = which(CLAUDE_BINARY)
    if binary is None:
        raise JudgeUnavailable(f"the '{CLAUDE_BINARY}' CLI is not on PATH")
    cmd = judge_command(binary, model=model, budget_usd=budget_usd)
    try:
        completed = runner(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as e:
        raise JudgeUnavailable(f"claude timed out after {timeout:g}s") from e
    except OSError as e:
        raise JudgeUnavailable(f"claude could not be started: {e}") from e
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:200]
        raise JudgeUnavailable(f"claude exited with {completed.returncode}: {detail}")
    try:
        envelope = json.loads(completed.stdout)
    except ValueError as e:
        raise JudgeUnavailable("claude did not return JSON") from e
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        detail = envelope.get("result") if isinstance(envelope, dict) else envelope
        raise JudgeUnavailable(f"claude reported an error: {str(detail)[:200]}")
    answer = envelope.get("structured_output")
    if answer is None:
        # Older CLIs only carry the JSON as text in "result".
        try:
            answer = json.loads(envelope.get("result") or "")
        except ValueError as e:
            raise JudgeUnavailable("claude returned no structured answer") from e
    matches = answer.get("matches") if isinstance(answer, dict) else None
    if not isinstance(matches, list):
        raise JudgeUnavailable("claude's answer has no 'matches' list")
    cost = envelope.get("total_cost_usd")
    if cost is not None:
        logger.info("claude judge call cost $%.3f", cost)
    return [m for m in matches if isinstance(m, dict)]


def anchors_from_matches(
    matches: Sequence[Mapping[str, Any]],
    windows: Sequence[tuple[int, int]],
    sub_cues: Sequence[JudgeCue],
    ref_cues: Sequence[JudgeCue],
    offsets: Sequence[float],
    *,
    min_offset: float,
    max_offset: float,
    consensus_seconds: float = ANCHOR_CONSENSUS_SECONDS,
) -> dict[int, float]:
    """Turn the judge's matches into ``{cue index: offset}`` anchors.

    The judge is not trusted blindly. A match is dropped when it names a cue
    that was not asked about or does not exist, when its offset is outside the
    search range, and when it stands alone: its offset agrees neither with the
    aligner's offset for that cue nor with the match before or after it.
    """
    asked = {i for first, last in windows for i in range(first, last + 1)}
    candidates: dict[int, float] = {}
    for match in matches:
        sub, ref = match.get("sub"), match.get("ref")
        if not isinstance(sub, int) or not isinstance(ref, int):
            continue
        if isinstance(sub, bool) or isinstance(ref, bool):
            continue
        i, j = sub - 1, ref - 1
        if i not in asked or not 0 <= j < len(ref_cues):
            continue
        offset = ref_cues[j].start - sub_cues[i].start
        if min_offset <= offset <= max_offset:
            candidates[i] = offset

    ordered = sorted(candidates)
    anchors: dict[int, float] = {}
    for pos, i in enumerate(ordered):
        offset = candidates[i]
        neighbours = [
            candidates[ordered[p]]
            for p in (pos - 1, pos + 1)
            if 0 <= p < len(ordered) and abs(ordered[p] - i) <= 2
        ]
        agrees = abs(offset - offsets[i]) <= consensus_seconds or any(
            abs(offset - other) <= consensus_seconds for other in neighbours
        )
        if agrees:
            anchors[i] = offset
    return anchors
