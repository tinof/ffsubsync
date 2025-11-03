# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FFsubsync is a language-agnostic command-line tool for automatic synchronization of subtitles with video. It uses voice activity detection (VAD) and Fast Fourier Transform (FFT) based signal processing to align subtitles with audio/video streams.

**Supported Platforms**: Linux and macOS only (no Windows support)

**Entry Points**: The tool provides three CLI aliases (all equivalent):
- `ffsubsync` - Main entry point
- `ffs` - Short alias
- `subsync` - Alternative alias

## Development Commands

### Environment Setup
```bash
# Install development dependencies (recommended)
pip install -e ".[dev]"

# Or separately
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks (required for development)
pip install pre-commit
pre-commit install
```

### Code Quality (Ruff)
```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Lint code (auto-fix where possible)
ruff check . --fix

# Format code
ruff format .

# Check formatting without changes
ruff format --check .
```

### Testing
```bash
# Run unit tests only (fast, recommended during development)
pytest -v -m 'not integration' tests/

# Run all tests including integration tests (requires test data)
pytest -v tests/

# Run with coverage
pytest --cov-config=.coveragerc --cov=ffsubsync tests/

# Run integration tests only
INTEGRATION=1 pytest -v -m 'integration' tests/
```

### Type Checking
```bash
# Run mypy type checker
mypy ffsubsync
```

### Legacy Make Commands
```bash
make clean        # Clean build artifacts
make lint         # Run flake8 (deprecated, use ruff)
make typecheck    # Run mypy
```

## Architecture

### Core Algorithm (3-step synchronization)

1. **Discretization**: Both video audio and subtitles are discretized into 10ms windows
2. **Speech Detection**:
   - Subtitles: Check if any subtitle is "on" during each window
   - Audio: Use VAD (WebRTC or Auditok) to detect speech presence
3. **Alignment**: Use FFT-based convolution (O(n log n)) to find optimal alignment between binary strings

### Key Components

**Pipeline Architecture** (inspired by scikit-learn):
- `sklearn_shim.py`: Custom `Pipeline` and `TransformerMixin` implementation (no sklearn dependency)
- Transformers follow fit/transform pattern for processing data through stages

**Core Modules**:
- `ffsubsync.py`: Main entry point, CLI argument parsing, orchestration
- `aligners.py`: FFT-based and max-score alignment algorithms
  - `FFTAligner`: Fast convolution-based alignment using numpy FFT
  - `MaxScoreAligner`: Evaluates multiple framerate ratios to find best alignment
- `speech_transformers.py`: Video/audio speech extraction transformers
  - `VideoSpeechTransformer`: Extract speech from video using ffmpeg + VAD
  - `SubtitleSpeechTransformer`: Convert subtitles to speech timeline
  - `DeserializeSpeechTransformer`: Load pre-serialized speech data
- `subtitle_transformers.py`: Subtitle manipulation transformers
  - `SubtitleScaler`: Scale subtitle timing by framerate ratio
  - `SubtitleShifter`: Apply time offset to subtitles
  - `SubtitleMerger`: Merge multiple subtitle streams
- `subtitle_parser.py`: Parse various subtitle formats (SRT, SSA/ASS via pysubs2)
- `generic_subtitles.py`: `GenericSubtitle` wrapper for uniform subtitle handling
- `ffmpeg_utils.py`: FFmpeg integration utilities
- `golden_section_search.py`: Golden-section search for optimal framerate ratio
- `constants.py`: Default values and configuration constants

**Processing Pipeline Example**:
```
subtitle file -> SubtitleParser -> SubtitleScaler -> SubtitleSpeechTransformer -> binary string
video file -> VideoSpeechTransformer -> binary string
binary strings -> FFTAligner -> time offset -> SubtitleShifter -> synchronized output
```

### VAD (Voice Activity Detection)

Backends supported:
- `webrtcvad-wheels` (default path: `--vad=webrtc`): Voice-specific detection
- `auditok` (`--vad=auditok`) [optional extra]: Energy/audio activity detector. Install with `pip install ffsubsync[auditok]`. Note: we keep `auditok<0.3.0` to avoid the newer PyAudio/portaudio build requirement on CI.
- `TEN VAD` (`--vad=tenvad` or `--vad=subs_then_tenvad`) [optional extra]: Low-latency, lightweight, high-accuracy streaming VAD. Requires 16 kHz audio; ffsubsync auto-sets `--frame-rate` to 16000 when selected. Install with `pip install ffsubsync[tenvad]`. If TEN VAD is not installed, the code falls back to WebRTC.

## Code Quality Standards

**Formatter & Linter**: Ruff (configured in `pyproject.toml`)
- Line length: 88 characters (Black-compatible)
- Target Python: 3.9+ (requires Python >= 3.9)
- Pre-commit hooks enforce Ruff checks automatically

**Type Checking**: mypy with type hints preferred

**Testing**: pytest with markers for unit (`-m 'not integration'`) and integration tests

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. **Code Quality**: Ruff linting and formatting (blocking)
2. **pipx Installation Test**: Linux/macOS, Python 3.9-3.13
3. **Unit Tests**: Linux/macOS, Python 3.9-3.13
4. **Integration Tests**: Ubuntu only, Python 3.10-3.11

All stages must pass for PR approval.

## Dependency Management

**Runtime Dependencies** (`pyproject.toml`):
- `ffmpeg-python`: FFmpeg wrapper for audio extraction
- `numpy`: FFT computations and array processing
- `srt`, `pysubs2`: Subtitle format parsing
- `webrtcvad-wheels`: Voice activity detection (default)
- Optional VAD backends: `auditok` (install with extra `auditok`), `ten-vad` (install with extra `tenvad`)
- `rich`, `tqdm`: CLI output formatting
- `chardet`, `charset_normalizer`, `faust-cchardet`: Character encoding detection

**Dev Dependencies**: black, flake8, mypy, pytest, pytest-cov, ruff, twine

**External Requirement**: ffmpeg must be installed on system (`brew install ffmpeg` on macOS)

## Important Implementation Details

- **No sklearn dependency**: Custom `Pipeline` implementation in `sklearn_shim.py` to avoid heavy dependencies
- **Encoding detection**: Uses multiple libraries (faust-cchardet, chardet, charset_normalizer) for robust subtitle encoding detection
- **Framerate handling**: Can sync subtitles with different framerates using `--gss` (golden-section search) or default ratio checks
- **Serialization**: Can serialize/deserialize speech data (`.npz` files) to skip expensive audio extraction
- **Platform-specific**: Unix-only design, uses POSIX-specific features

## Common Development Workflows

### Adding a new VAD backend
1. Add detector function in `speech_transformers.py` (follow `_make_auditok_detector` pattern)
2. Register in `VideoSpeechTransformer.__init__` VAD selection logic
3. Add to `constants.py` if needed
4. Update tests in `tests/`

### Supporting a new subtitle format
1. Extend `subtitle_parser.py` with new parser
2. Update `GenericSubtitle` in `generic_subtitles.py` if format requires new handling
3. Add format to `SUBTITLE_EXTENSIONS` in `constants.py`
4. Add tests in `tests/test_subtitles.py`

### Debugging sync failures
- Use `--make-test-case` to create test archive with debug data
- Check `--max-offset-seconds` if offset > 60s
- Try `--no-fix-framerate` to assume identical framerates
- Try `--gss` for exhaustive framerate ratio search
- Try `--vad=auditok` for low-quality audio

## Testing Notes

- **Unit tests**: Fast, no external dependencies, run on every commit
- **Integration tests**: Require test data files, slower, run on Ubuntu only in CI
- **Markers**: Use `@pytest.mark.integration` for integration tests
- Integration tests require `INTEGRATION=1` environment variable
