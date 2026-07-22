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

Python support starts at 3.10 (`requires-python = ">=3.10"`). Packaging uses
setuptools and Versioneer.

## CLI Entry Points

`pyproject.toml` installs four console scripts:

- `ffsubsync`, `ffs`, and `subsync`: equivalent low-level CLIs implemented by
  `ffsubsync:main`. These expect explicit reference/input/output arguments.
- `ssync`: convenience workflow implemented by `ffsubsync.ssync:main`.

`ssync` is a first-class workflow:

- `ssync episode.mkv` syncs the matching language subtitle beside that video in
  place.
- Plain `ssync` scans the current directory recursively for videos and processes
  them in sorted order.
- The default subtitle language is Finnish (`fin`), with `fin`/`fi` aliases.
- `--lang` changes the suffix used for subtitle discovery.
- `--preflight` is passed through to the sync engine.
- `--reference-source audio` is the default and does not probe embedded subtitle
  streams.
- `--reference-source embedded` opts into selecting and extracting an embedded
  subtitle reference, preferring non-target English streams and falling back to
  audio if extraction fails.
- `--dry-run` reports resolved jobs and reference policy without running
  extraction or synchronization.

On this Ubuntu machine, prior local deployment used a `pipxu` managed install and
`/home/ubuntu/bin/ssync` is a user-facing wrapper. If the user asks to install or
refresh the command they actually run, verify `ssync` from outside the checkout
so imports do not accidentally come from the repo.

## Development Commands

Install the project for local development:

```bash
pip install -e ".[dev]"
```

Alternative legacy setup:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

Install hooks:

```bash
pip install pre-commit
pre-commit install
```

Lint and format:

```bash
ruff check .
ruff check . --fix
ruff format .
ruff format --check .
pre-commit run --all-files
pre-commit run --all-files --hook-stage push
```

Tests:

```bash
pytest -v -m 'not integration' tests/
pytest -v tests/
pytest --cov-config=.coveragerc --cov=ffsubsync tests/
INTEGRATION=1 pytest -v -m 'integration' tests/
```

Type checking:

```bash
mypy ffsubsync
```

Legacy Make targets still exist:

```bash
make clean
make lint
make typecheck
```

Use focused commands while developing. Examples:

```bash
uv run pytest -q tests/test_ssync.py
uv run pytest -q tests/test_misc.py
ruff check ffsubsync/ssync.py tests/test_ssync.py
ruff format --check ffsubsync/ssync.py tests/test_ssync.py
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

- `pip install ffsubsync[tenvad]`: native TEN-VAD. Linux x64 and macOS.
- `pip install ffsubsync[tenvad-onnx]`: ONNX Runtime backend, including Linux
  ARM64 support.

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
- `tests/test_integration.py`: integration scenarios gated by `INTEGRATION=1`.

For changes to `ssync`, prefer tests against pure workflow functions and an
injected executor instead of monkeypatching `sys.argv` or the global `run`
function.

## Code Quality And Style

Ruff is the formatter and linter. Configuration is in `pyproject.toml`:

- line length: 88
- target: Python 3.10
- selected lint families: `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `SIM`, `RUF`
- notable ignores: `E501`, `E722`, `B008`

Black and flake8 remain in development dependencies for legacy workflows, but
Ruff is authoritative for current formatting and linting.

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

GitHub Actions runs:

1. Ruff and pre-commit quality checks on Ubuntu with Python 3.11.
2. pipx installation checks on Ubuntu and macOS for Python 3.10 through 3.13.
3. Unit tests on Ubuntu and macOS for Python 3.10 through 3.13.
4. Integration tests on Ubuntu for Python 3.10 and 3.11.

CI verifies `ffsubsync`, `ffs`, and `subsync` help output in the pipx job. It
does not currently verify `ssync` there, so local `ssync` workflow tests matter.
