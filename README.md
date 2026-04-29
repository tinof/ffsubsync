FFsubsync
=======

[![CI Status](https://github.com/smacke/ffsubsync/workflows/ffsubsync/badge.svg)](https://github.com/smacke/ffsubsync/actions)
[![Support Ukraine](https://badgen.net/badge/support/UKRAINE/?color=0057B8&labelColor=FFD700)](https://github.com/vshymanskyy/StandWithUkraine/blob/main/docs/README.md)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)
[![Documentation Status](https://readthedocs.org/projects/ffsubsync/badge/?version=latest)](https://ffsubsync.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/ffsubsync.svg)](https://pypi.org/project/ffsubsync)

> **This is a fork of [smacke/ffsubsync](https://github.com/smacke/ffsubsync)** with significant reliability improvements and convenience features added on top of the original. The sections below describe what makes this fork different; the rest of the document covers the tool as a whole.

---

## What This Fork Adds

### Robust Alignment Engine (drift fix)

The original FFT alignment works well for straightforward sync cases, but fails silently on difficult inputs — particularly when the segmented aligner produces noisy, uncorrelated per-window offsets, or when the golden-section framerate search converges to a physically impossible ratio. This fork adds three layers of defence that together prevent the "synced at start, drifts to the end" failure mode:

**Segmented aligner confidence gate** (`aligners.py`): The `SegmentedAligner` now requires a strict majority (>50%) of its sliding windows to agree on the offset. If no majority is reached, it raises `FailedToFindAlignmentException` and the caller falls back rather than silently accepting a noise-derived result.

**Framerate plausibility snap** (`aligners.py`, `constants.py`): After golden-section search (GSS) finds an optimal framerate ratio, the result is checked against all known physical framerate pairs (24/23.976, 25/24, 25/23.976, and their reciprocals). If the GSS ratio is more than 0.5% from the nearest known pair, it is discarded. This prevents non-physical ratios like 0.986 (which produce ~18 s of end-of-file drift over a 22-minute episode) from ever reaching the output.

**Strategy selector margin** (`ffsubsync.py`): When both a primary and an adaptive alignment strategy are tried, the adaptive result must exceed the primary score by at least 15% to override it. Without this guard, a small numerical edge from an incomparable score metric was enough to flip the selection.

**Adaptive skip gate — drift check** (`ffsubsync.py`): After primary alignment completes, a cheap two-halves consistency check (`_primary_has_no_drift`) splits reference and subtitle speech in half and runs `FFTAligner` on each half independently. If both halves agree on the same offset within 0.5 s, the subtitle has no mid-file drift and the expensive adaptive (segmented) strategy is skipped entirely. Additionally, when adaptive *does* run, it now uses only the framerate ratio chosen by primary instead of re-scanning all known ratios and running a full GSS sweep — dropping adaptive cost from ~63 FFT operations to ~9.

### `ssync` — Language-Aware Convenience Wrapper

`ssync` is a single-argument wrapper that finds the subtitle file next to a video automatically and syncs it in-place. You don't need to pass `-i` / `-o` paths.

```bash
# Default: finds <video>.fin.srt and syncs it against the video
ssync "The Collapse (2019) - S01E03 - The Airfield D+6 [Bluray-1080p].mkv"

# Different language suffix
ssync --lang eng "Episode.mkv"

# Preview resolved paths without running sync
ssync --dry-run "Episode.mkv"
```

### `--preflight` — Fast "Already in Sync?" Check

Passing `--preflight` (or `--skip-if-synced`) runs a short probe on the first ~2 minutes of audio before committing to the full extraction pipeline. If the subtitle is already aligned, the full pipeline is skipped entirely and the file is written with a zero shift.

The probe passes three gates: offset ≤ 0.5 s, score margin ≥ 1.5x, and at least 5% voiced frames in the window (so silent intros don't produce false positives). Any gate failing causes preflight to abstain and fall through to the normal pipeline.

```bash
# With ssync
ssync --preflight "Episode.mkv"

# With the low-level CLI
ffs video.mkv -i subtitle.srt -o subtitle.srt --preflight
```

### Piecewise Sync for Mid-File Drift

When subtitles drift progressively through the file (different video edits, commercial breaks removed differently, or translations from a different release), a single global offset cannot fix the problem. The piecewise sync tool applies a three-phase algorithm:

1. **Ratio-based time warping** — maps source timeline proportionally to reference
2. **Local offset correction** — finds optimal offset per 60-second window using median filtering
3. **Smooth interpolation** — blends corrections across window boundaries

```bash
# Sync Finnish subtitles using French reference (both .srt)
python -m ffsubsync.tools.piecewise_sync french.srt finnish.srt synced_finnish.srt --verify

# Custom window size (30 seconds)
python -m ffsubsync.tools.piecewise_sync --window 30000 reference.srt input.srt output.srt
```

### ARM64 Support via ONNX TEN-VAD

The native `ten-vad` package ships prebuilt binaries only for Linux x64 and macOS. This fork adds an ONNX Runtime-based backend that exposes the same interface on all platforms, including ARM64 Linux (Oracle Cloud, AWS Graviton, Raspberry Pi).

- Install with `pip install ffsubsync[tenvad-onnx]` for all platforms
- Install with `pip install ffsubsync[tenvad]` for Linux x64 / macOS (native, slightly faster)
- Falls back automatically: native TEN-VAD → ONNX TEN-VAD → WebRTC

---

A command-line tool for language-agnostic automatic synchronization of subtitles with video, so that
subtitles are aligned to the correct starting point within the video.

Turn this:                       |  Into this:
:-------------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/smacke/ffsubsync/master/resources/img/tearing-me-apart-wrong.gif)  |  ![](https://raw.githubusercontent.com/smacke/ffsubsync/master/resources/img/tearing-me-apart-correct.gif)

Helping Development
-------------------
Please consider [supporting Ukraine](https://github.com/vshymanskyy/StandWithUkraine/blob/main/docs/README.md)
rather than donating directly to this project. That said, at the request of
some, you can now help cover my coffee expenses using the Github Sponsors
button at the top, or using the below Paypal Donate button:

[![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=XJC5ANLMYECJE)

Install
-------
First, make sure ffmpeg is installed:

**On macOS:**
~~~
brew install ffmpeg
~~~

**On Linux (Ubuntu/Debian):**
~~~
sudo apt update
sudo apt install ffmpeg
~~~

**On Linux (CentOS/RHEL/Fedora):**
~~~
sudo dnf install ffmpeg
~~~

### Recommended Installation (pipx)

The recommended way to install ffsubsync is using [pipx](https://pypa.github.io/pipx/), which installs the package in an isolated environment and makes the CLI commands globally available.

**ARM64/aarch64 (ONNX backend — all platforms):**

~~~
pipx install "ffsubsync[tenvad-onnx] @ git+https://github.com/tinof/ffsubsync@LATEST"
~~~

**Linux x64 / macOS (native TEN VAD — best performance):**

~~~
pipx install "ffsubsync[tenvad] @ git+https://github.com/tinof/ffsubsync@LATEST"
~~~

**WebRTC only (no TEN VAD):**

~~~
pipx install "git+https://github.com/tinof/ffsubsync.git"
~~~

**Development head (master branch):**

~~~
pipx install 'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad-onnx]'
~~~

If you already installed without the extra, you can add TEN VAD later:

~~~
# ARM64:
pipx inject ffsubsync onnxruntime

# Linux x64 / macOS:
pipx inject ffsubsync 'ten-vad'
~~~

### Alternative Installation (pip)

~~~bash
git clone https://github.com/tinof/ffsubsync.git
cd ffsubsync

# ARM64 systems
pip install ".[tenvad-onnx]"

# Linux x64 / macOS
pip install ".[tenvad]"

# WebRTC only
pip install .
~~~

> **Note on TEN VAD wheels**: on Debian/Ubuntu you may need `sudo apt install libc++1`. If `ten-vad` fails to build on your platform, install without the extra (defaults to WebRTC VAD) or use the ONNX backend instead.

**VAD Backend Priority (selected automatically):**
1. `ten-vad` installed → native TEN VAD (best performance, Linux x64/macOS)
2. `onnxruntime` installed → ONNX TEN VAD (ARM64 compatible)
3. Otherwise → WebRTC VAD (always available)

Usage
-----
After installation, four CLI commands are available:

- `ffs`, `subsync`, `ffsubsync` — equivalent low-level CLI (pass full paths explicitly)
- `ssync` — convenience wrapper that auto-discovers subtitle files by language suffix

Supported subtitle formats: `.srt`, `.ass/.ssa`, MicroDVD `.sub`, and `.vtt`. MicroDVD frame rates are preserved when converting back to `.sub`.

**Basic synchronization with video:**
~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

**Synchronization using reference subtitles:**
~~~
ffs reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

**Running as a Python module:**
~~~
python -m ffsubsync video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

The tool uses the file extension to decide whether to perform voice activity
detection on the audio or to directly extract speech from an srt file.

### `ssync` convenience command

`ssync` takes only the video path, auto-finds the matching subtitle file in the same directory, and syncs it in-place.

Default behavior:
- Looks for `<video_basename>.fin.srt` (case-insensitive language suffix matching)
- Syncs subtitle against the video
- Writes output back to the same subtitle file

~~~bash
# Default language suffix: .fin.srt
ssync "The Family Next Door (AU) (2025) - S01E04 - Fran [WEBRip-1080p].mkv"

# Different language suffix
ssync --lang eng "Episode.mkv"

# Preview resolved paths only
ssync --dry-run "Episode.mkv"

# Skip full pipeline if subtitles appear already in sync
ssync --preflight "Episode.mkv"
~~~

Always quote filenames that contain spaces or parentheses.

Platform Support
----------------
- **Linux** (all major distributions)
- **macOS** (Intel and Apple Silicon)

Windows is not supported. The tool requires Unix-like systems.

Requirements
------------
- **Python:** 3.10 or higher
- **ffmpeg:** Must be installed and available in your system PATH
- **Dependencies:** All Python dependencies are installed automatically

Sync Issues
-----------
If the sync fails, the following recourses are available:
- Try `--no-fix-framerate` to assume identical video / subtitle framerates
- Try `--gss` to use [golden-section search](https://en.wikipedia.org/wiki/Golden-section_search) for the optimal framerate ratio (by default, only common ratios are evaluated)
- Try a larger `--max-offset-seconds` (default: 60) if the subtitles are very far out of sync
- Try `--vad=tenvad` for higher-accuracy speech detection (requires the `tenvad` or `tenvad-onnx` extra)

For progressive mid-file drift that survives the above flags, use the **Piecewise Sync** tool described in the [What This Fork Adds](#what-this-fork-adds) section above.

If sync still fails, similar tools worth trying:
- [sc0ty/subsync](https://github.com/sc0ty/subsync): speech-to-text + word morpheme matching
- [kaegi/alass](https://github.com/kaegi/alass): Rust-based subtitle synchronizer with dynamic programming
- [tympanix/subsync](https://github.com/tympanix/subsync): neural net approach
- [oseiskar/autosubsync](https://github.com/oseiskar/autosubsync): spectrogram + logistic regression
- [pums974/srtsync](https://github.com/pums974/srtsync): similar approach (WebRTC + FFT)

Speed
-----
`ffsubsync` usually finishes in 20 to 30 seconds, depending on the length of
the video. The most expensive step is raw audio extraction. If you already have
a correctly synchronized reference srt file (so audio extraction can be skipped),
ffsubsync typically runs in under a second.

With `--preflight`, already-synced files are detected in a few seconds without
running the full pipeline.

VAD Benchmarks
--------------
Performance comparison of different synchronization methods:

| Method | Time | Score | Avg Error |
|--------|------|-------|-----------|
| sub-to-sub | 1.8s | 81,040 | 0.23s |
| audio-webrtc | 30.8s | 62,669 | 0.11s |
| audio-tenvad | 30.5s | 62,669 | 0.11s |

WebRTC and TEN-VAD have identical accuracy. Sub-to-sub (using a reference subtitle
file) is fastest but slightly less accurate than audio-based synchronization.

How It Works
------------
The synchronization algorithm operates in 3 steps:
1. Discretize both the video file's audio stream and the subtitles into 10ms
   windows.
2. For each 10ms window, determine whether that window contains speech. This is
   trivial for subtitles (any subtitle "on" during the window counts). For
   audio, the default VAD is WebRTC; pass `--vad=tenvad` or
   `--vad=subs_then_tenvad` to use TEN VAD instead (requires the `tenvad` or
   `tenvad-onnx` extra).
3. Align the resulting binary strings using FFT-based convolution (O(n log n))
   to find the offset that maximises matched speech frames.

Limitations
-----------
The `ffs` command applies a **single global offset and scale factor**, which works
well when subtitles are consistently offset throughout the file or the framerate
mismatch is uniform. It may fail for:
- Mid-file drift (different video edits, commercial breaks removed differently)
- Translations with very different segmentation (lines split/combined differently)
- Subtitles from a completely different source release

Use the **Piecewise Sync** tool for these cases:
```bash
python -m ffsubsync.tools.piecewise_sync reference.srt input.srt output.srt --verify
```

History
-------
The implementation for this project was started during HackIllinois 2019, for
which it received an **_Honorable Mention_** (ranked in the top 5 projects,
excluding projects that won company-specific prizes).

Credits
-------
This project would not be possible without the following libraries:
- [ffmpeg](https://www.ffmpeg.org/) and the [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) wrapper, for extracting raw audio from video
- VAD from [webrtc](https://webrtc.org/) and the [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) wrapper, for speech detection
- [TEN VAD](https://github.com/TEN-framework/ten-vad) for high-accuracy, low-latency voice activity detection
- [srt](https://pypi.org/project/srt/) for operating on [SRT files](https://en.wikipedia.org/wiki/SubRip#SubRip_text_file_format)
- [numpy](http://www.numpy.org/) and, indirectly, [FFTPACK](https://www.netlib.org/fftpack/), which powers the FFT-based algorithm for fast scoring of alignments between subtitles (or subtitles and video)
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html), [rich](https://github.com/willmcgugan/rich), and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
Code in this project is [MIT licensed](https://opensource.org/licenses/MIT).
