# AGENTS.md

This is the canonical AI-agent guide for this repository. `CLAUDE.md` should
link here instead of carrying a separate copy.

## Project Overview

FFsubsync is a Python command-line tool for synchronizing subtitle timing with a
reference video, audio track, subtitle file, or serialized speech array. It is
language-agnostic: both audio/video and subtitles are converted into speech or
non-speech timelines, then aligned with FFT-based signal processing.

Supported platforms are Linux and macOS. Windows is not supported. The external
`ffmpeg` and `ffprobe` binaries must be installed and available on `PATH` unless
the user passes an explicit ffmpeg path.

Python support starts at 3.11 (`requires-python = ">=3.11"`). The project is
managed with uv and follows the simple-modern-uv template (flat `ffsubsync/`
layout kept, not Copier-managed). The build backend is hatchling; the version
comes from git tags through uv-dynamic-versioning (tags may or may not have a
`v` prefix), and `ffsubsync/version.py` reads it from the installed package
metadata. `uv.lock` is committed, and `uv.toml` sets a 14-day `exclude-newer`
cool-off. This fork is not published to PyPI.

## CLI Entry Points

`pyproject.toml` installs four console scripts:

- `ffsubsync`, `ffs`, and `subsync`: equivalent low-level CLIs implemented by
  `ffsubsync:main`. These expect explicit reference/input/output arguments.
- `ssync`: convenience workflow implemented by `ffsubsync.ssync:main`.
- `piecewise-sync`: standalone piecewise drift tool implemented by
  `ffsubsync.tools.piecewise_sync:main` (also runnable as
  `python -m ffsubsync.tools.piecewise_sync`).

`ssync` is a first-class workflow:

- `ssync episode.mkv` syncs the matching language subtitle beside that video in
  place.
- Plain `ssync` scans the current directory recursively for videos and processes
  them in sorted order.
- The default subtitle language is Finnish (`fin`), with `fin`/`fi` aliases.
- Subtitle candidate discovery searches in order: target language `.srt`, bare `<stem>.srt`, target language `.sub` (converted to `.srt` via ffmpeg ISO-8859-15, deleting the source `.sub` on success), fallback language `.srt`, and fallback language `.sub`. Matching is case-insensitive via a directory index (`_case_insensitive_index()`), so `.fin.srt`, `.FIN.srt`, and `.Fin.srt` all resolve on case-sensitive filesystems; the real on-disk path is what gets rewritten in place.
- `--lang` changes the suffix used for subtitle discovery.
- `--fallback-lang` changes the fallback language suffix used when no target-language subtitle is found (default: `en`). Setting `--fallback-lang ""` disables the fallback pass.
- `--preflight` is passed through to the sync engine.
- `--reference-source audio` is the default and does not probe embedded subtitle
  streams.
- `--reference-source embedded` opts into selecting an embedded subtitle
  reference. `_pick_reference_subtitle_stream()` ranks a text stream in another
  language (English first), then a PGS stream in another language, then a
  target-language stream. VobSub/DVB streams are never used. A text stream is
  extracted to `.srt`; a PGS stream is not extracted: ssync passes it to the
  engine as `--pgs-ref-stream 0:<index>`, and `PGSSpeechTransformer` reads the
  caption timings from the container packets with ffprobe (caption end = next
  "clear" packet when the muxer stores no duration; the container start time is
  subtracted). PGS is only offered for `.mkv` files, because in a Blu-ray
  transport stream each PGS segment may be its own packet. Falls back to audio
  if there is no usable stream, text extraction fails, or the PGS reference
  raises. A job is reported as failed when the engine's result has
  `sync_was_successful: False` (the engine's own exit code stays 0).
- `--dry-run` reports resolved jobs and reference policy without running
  extraction or synchronization.
- Advanced tuning flags are forwarded to the sync engine for when the default
  sync is off: `--gss`, `--vad`, `--max-offset-seconds`,
  `--use-segmented-aligner`, `--no-fix-framerate`, and `--no-auto-sync`. An
  explicit `--vad` overrides the `webrtc` VAD that ssync forces for audio
  references.
- The low-quality safety net is on by default: `build_sync_args()` sets the
  engine's `--skip-sync-on-low-quality`. When `assess_alignment_quality()` in
  `ffsubsync.py` returns reasons (negative score, offset above
  `--quality-max-offset-seconds`, default 60 s and raised to match an explicit
  `--max-offset-seconds`, framerate scale deviation above 0.1, or a scale that
  is not 1.0 or a known ratio within `FRAMERATE_SNAP_TOLERANCE`, via
  `aligners.nearest_known_framerate_ratio()`), `try_sync()`
  writes the original subtitle, sets `result["kept_original_reason"]`, and skips
  piecewise. ssync reports `kept_original` with exit code 1.
  `--no-quality-gate` disables it. The text-embedded piecewise path
  (`tools/piecewise_sync`) does not go through the engine and has no gate.
- `--piecewise` corrects progressive mid-file drift that a single offset and
  scale cannot fix. Against the **audio** by default (`ffsubsync/piecewise.py`
  measures residual offsets in overlapping windows of the cached speech
  timeline and warps cue timings between the resulting anchors, using no extra
  ffmpeg pass). With `--reference-source embedded` it instead runs
  `ffsubsync.tools.piecewise_sync` against an extracted embedded subtitle
  stream, and skips a video (exit code unchanged) when no embedded stream is
  available; `--piecewise-window` (milliseconds) applies only to that path.
  When the picked stream is PGS there is no text to warp, so ssync runs the
  engine's audio-style piecewise against the PGS timings instead.
- `--piecewise-mode split` (with `--piecewise`) runs the engine's split-penalty
  aligner (`--split-penalty`, ssync default 30 s) instead of drift anchors, for
  discrete jumps. It always goes through the engine, also with an embedded text
  reference (the extracted `.srt` is the reference). Drift mode cannot fix a
  jump larger than `DEFAULT_MAX_RESIDUAL_OFFSET_SECONDS` (15 s); split mode
  searches +-`--max-offset-seconds` around the global offset.
- `--ai` is for a subtitle made for another cut of the video (added scenes, many
  forward jumps). `_execute_ai_job()` extracts an embedded **text** stream
  (`_pick_ai_reference_stream()`; PGS or no stream -> `skipped`), runs
  `cut_aligner.compute_cut_offsets()`, sends the cues of
  `cut_aligner.review_windows()` to the judge, turns its matches into anchors
  (`ai_judge.anchors_from_matches()`), re-runs the aligner (at most
  `AI_MAX_JUDGE_ROUNDS`), and applies the offsets with `VariableSubtitleShifter`.
  It does not call the engine. The gate is `score_per_cue <
  AI_MIN_SCORE_PER_CUE` -> `kept_original`, exit code 1. `--ai-no-judge` skips
  Claude, and so does a first alignment below the gate. `--ai-fallback` runs
  AI mode after the normal sync ends `failed` or `kept_original`, or ends
  `synced` with `_embedded_agreement()` below `AI_FALLBACK_MIN_AGREEMENT` (the
  engine's gate passes some wrong framerate scales). `execute_job()` restores
  the original bytes first, because the engine writes in place, and puts the
  engine's output back when AI mode does not sync.
  `--ai-model`, `--ai-timeout` and `--ai-budget-usd` go to the `claude` call.
  `--ai` excludes `--piecewise` and `--ai-fallback`. The judge is injected
  (`main(..., judge=...)`, `execute_job(..., judge=...)`) like the executor.

On this Ubuntu machine the commands are an editable uv tool install of this
checkout (`uv tool install --editable /home/ubuntu/ffsubsync --with onnxruntime`,
commands in `~/.local/bin`), so code changes are live at once. Run the install
again with `--force` after a change to dependencies or entry points. If the user
asks to install or refresh the command they actually run, verify `ssync` from
outside the checkout.

## Development Commands

Install the project and all development dependencies into `.venv`:

```bash
make install        # uv sync --all-groups --extra tenvad-onnx
```

Lint and format (codespell, ruff check, ruff format):

```bash
make lint           # fixes files
make lint-check     # check only, same as CI
uv run ruff check .
uv run ruff format --check .
```

Tests:

```bash
make test                       # uv run pytest -m 'not integration'
uv run pytest -v tests/
uv run pytest --cov-config=.coveragerc --cov=ffsubsync tests/
make test-integration           # INTEGRATION=1 uv run pytest -m integration
```

Type checking (basedpyright, not a CI gate; the legacy code has known errors):

```bash
make typecheck
```

Dependencies and builds:

```bash
uv add <name>           # runtime dependency
uv add --dev <name>     # development dependency
make upgrade            # upgrade the lock file
make build              # sdist and wheel in dist/
```

After a dependency change, commit `uv.lock`; CI installs with `uv sync --locked`.
The `tenvad` extra is a git dependency without Linux ARM64 support, so do not
use `--all-extras` on this machine.

Use focused commands while developing. Examples:

```bash
uv run pytest -q tests/test_ssync.py
uv run pytest -q tests/test_misc.py
uv run ruff check ffsubsync/ssync.py tests/test_ssync.py
uv run ruff format --check ffsubsync/ssync.py tests/test_ssync.py
```

## Core Algorithm

The normal synchronization pipeline is:

1. Discretize reference and subtitle speech into 10 ms windows
   (`SAMPLE_RATE = 100`).
2. Detect speech:
   - Subtitles are parsed into active or inactive windows.
   - Audio/video references are extracted with ffmpeg and a VAD backend.
3. Align the binary timelines with FFT-based convolution.
4. Shift and optionally scale subtitle timings, then write the output file.

Framerate mismatch handling is built into the alignment layer. Known ratios live
in `constants.py` as `FRAMERATE_RATIOS`; golden-section search candidates are
snapped only if they are within `FRAMERATE_SNAP_TOLERANCE = 0.005` of a known
physical ratio.

## Important Modules

- `ffsubsync/ffsubsync.py`: low-level CLI parser and top-level sync
  orchestration. `run()` validates args and delegates to `_run_impl()`.
- `ffsubsync/ssync.py`: high-level `ssync` workflow. Key types are
  `SsyncOptions`, `SsyncJob`, `SsyncSyncRequest`, and `SsyncResult`.
- `ffsubsync/aligners.py`: `FFTAligner`, `SegmentedAligner`, and
  `MaxScoreAligner`.
- `ffsubsync/speech_transformers.py`: audio/video/subtitle speech extraction.
- `ffsubsync/subtitle_transformers.py`: scaling, shifting, and merging subtitle
  streams.
- `ffsubsync/subtitle_parser.py`: SRT and SSA/ASS parsing through `srt` and
  `pysubs2`.
- `ffsubsync/generic_subtitles.py`: common subtitle abstraction.
- `ffsubsync/preflight.py`: fast already-synced check used by `--preflight`.
- `ffsubsync/piecewise.py`: audio-based drift correction. `compute_window_offsets()`
  measures residual offsets per window on the 100 Hz binary speech arrays;
  `build_anchors()` filters them (rank-based score gate, median-filter outlier
  rejection, monotonicity) into anchors applied by `PiecewiseSubtitleShifter`.
  Wired into the engine by `--piecewise-audio` and into `ssync` by `--piecewise`.
  Note FFT scores are unnormalized and often negative, so score gating must be
  rank-based, never relative to a mean or median.
- `ffsubsync/split_aligner.py`: alass-style split-penalty DP (ported from
  upstream). `compute_split_offsets()` returns one offset per cue plus the DP
  objective; `enforce_cue_order()` moves orphan cues at negative jumps after the
  previous segment. The engine helper `_compute_split_offsets()` in
  `ffsubsync.py` centers the search on each framerate candidate's own global FFT
  offset and keeps the scale with the best DP objective; `VariableSubtitleShifter`
  applies the result. Upstream's default penalty (5 s) splits correct files
  against a real VAD reference; the fork's default is 30 s (see `constants.py`).
- `ffsubsync/cut_aligner.py`: cut-aware aligner behind `ssync --ai`. Works on
  cue timings of the subtitle and of a reference *text* track. The score is a
  kernel around reference cue starts plus a small overlap term; the Viterbi
  path may stay, jump forward (`jump_penalty`), or step back at most
  `max_back_step`. `anchors` restrict a cue to +-2.5 s around a known offset.
  `is_dialogue_text()` drops SDH sound descriptions. Overlap-only scoring (as
  in `split_aligner`) fails on this problem; do not replace the start kernel.
- `ffsubsync/ai_judge.py`: prompt building, the headless `claude -p` call
  (`--tools ""`, `--setting-sources ""`, `--json-schema`, answer in
  `structured_output`), and validation of the returned matches. Claude returns
  cue pairs, never offsets. Every failure raises `JudgeUnavailable`.
- `ffsubsync/ten_vad_onnx.py` and `ffsubsync/onnx_models/`: ONNX TEN-VAD
  compatibility backend.
- `ffsubsync/tools/piecewise_sync.py`: standalone tool for progressive mid-file
  drift. Run with `python -m ffsubsync.tools.piecewise_sync`.
- `ffsubsync/sklearn_shim.py`: small local `Pipeline` and `TransformerMixin`
  shim. The project intentionally avoids a scikit-learn dependency.

## Alignment Strategy Notes

`get_alignment_strategies()` builds a primary strategy from explicit CLI flags,
then optionally adds adaptive strategies. By default, adaptive sync may try a GSS
scale search and segmented voting unless the primary strategy already covers that
configuration. `--no-auto-sync` disables these extra strategies.

`_primary_has_no_drift()` is a cheap two-halves consistency check. If both halves
agree with the primary offset within 0.5 seconds, adaptive segmented alignment is
skipped.

`SegmentedAligner` performs sliding-window alignment with overlap. It uses a
strict majority gate: more than half of usable windows must agree on an offset
bin, otherwise it raises `FailedToFindAlignmentException`. It exposes
`confidence_` and `vote_ratio_` for diagnostics.

When working on late-from-start or long-file alignment bugs, the highest-signal
tests are usually `tests/test_segmented_aligner.py` and
`tests/test_strategy_selection.py`.

## VAD Backends

`DEFAULT_VAD` is `subs_then_webrtc`. For video/audio references this first tries
subtitle-stream based extraction where possible, then falls back to WebRTC audio
VAD.

Supported explicit VAD choices include:

- `webrtc`: WebRTC VAD through `webrtcvad-wheels`.
- `subs_then_webrtc`: subtitle-stream first, then WebRTC.
- `tenvad`: TEN-VAD native backend.
- `subs_then_tenvad`: subtitle-stream first, then TEN-VAD.
- `whisper`: Whisper-based speech extraction.

TEN-VAD requires 16 kHz audio; the code adjusts frame rate automatically for
TEN-VAD modes. Optional extras:

- `tenvad`: native TEN-VAD. Linux x64 and macOS.
- `tenvad-onnx`: ONNX Runtime backend, including Linux ARM64 support.
- `whisper`: faster-whisper, for `--vad whisper`.

If native `ten-vad` import fails, the code attempts the ONNX backend before
falling back to WebRTC.

## Testing Guide

Unit tests are designed to be fast and run without integration test data. Mark
slow external-data tests with `@pytest.mark.integration`; integration tests
require `INTEGRATION=1`.

Test files by area:

- `tests/test_ssync.py`: `ssync` path resolution, reference policy, and CLI
  workflow.
- `tests/test_misc.py`: parser and version helpers.
- `tests/test_alignment.py`: core aligner behavior.
- `tests/test_segmented_aligner.py`: segmented voting and tail-window behavior.
- `tests/test_strategy_selection.py`: adaptive strategy and framerate-ratio
  selection.
- `tests/test_tenvad_backend.py`: TEN-VAD backend selection and fallback.
- `tests/test_subtitles.py`: subtitle parsing and transformations.
- `tests/test_cut_aligner.py`, `tests/test_ai_judge.py`: AI mode. The regression
  fixture `tests/data/cut_blue_lights_s01e02.json` holds cue timings only.
  Claude is never called in tests; inject a fake judge or runner.
- `tests/test_integration.py`: integration scenarios gated by `INTEGRATION=1`.

For changes to `ssync`, prefer tests against pure workflow functions and an
injected executor instead of monkeypatching `sys.argv` or the global `run`
function.

## Code Quality And Style

Ruff is the formatter and linter. Configuration is in `pyproject.toml`:

- line length: 88
- target: Python 3.11
- selected lint families: `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `SIM`, `RUF`
- notable ignores: `E501`, `E722`, `B008`

codespell checks spelling. `devtools/lint.py` runs codespell and Ruff together.

Type hints are preferred, but the codebase is only partially typed and uses
numpy heavily. Do not introduce broad type-only refactors unless needed for the
task.

## Common Workflows

Adding a VAD backend:

1. Add a detector factory in `speech_transformers.py`.
2. Register selection and fallback behavior in `VideoSpeechTransformer`.
3. Add constants or CLI choices if needed.
4. Add focused tests, usually near `tests/test_tenvad_backend.py`.

Adding subtitle format support:

1. Extend parsing in `subtitle_parser.py`.
2. Update `GenericSubtitle` or related abstractions if needed.
3. Add the extension to `SUBTITLE_EXTENSIONS` in `constants.py`.
4. Add tests in `tests/test_subtitles.py`.

Debugging sync failures:

- Use `--make-test-case` to capture debug data.
- Increase `--max-offset-seconds` if the real offset may exceed 60 seconds.
- Try `--no-fix-framerate` when framerates are known to match.
- Try `--gss` for exhaustive framerate-ratio search.
- Try `--use-segmented-aligner` for long intros, sparse speech, or misleading
  global matches.
- Try `--vad=tenvad` when WebRTC speech detection appears weak.
- Use `--preflight` to skip full sync when subtitles are probably already
  aligned.
- Use `python -m ffsubsync.tools.piecewise_sync` for progressive mid-file drift.

## CI

GitHub Actions calls uv directly (`uv sync --locked`, pinned action SHAs):

1. codespell and Ruff (`devtools/lint.py --check`) on Ubuntu.
2. `uv tool install` of the built wheel on Ubuntu and macOS for Python 3.11
   through 3.14, with `--help` for all five commands, including `ssync`.
3. Unit tests on Ubuntu x64, Ubuntu ARM64 and macOS for Python 3.11 through 3.14.
4. Integration tests on Ubuntu for Python 3.11 and 3.12.

Python 3.14 jobs do not fail the workflow.
