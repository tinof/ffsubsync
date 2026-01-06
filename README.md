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

## 🎯 Fork Highlights: ARM64 Support with ONNX-based TEN VAD

This fork adds **ONNX Runtime-based TEN VAD** support, enabling high-accuracy voice activity detection on **ARM64/aarch64 platforms** (Oracle Cloud, AWS Graviton, Raspberry Pi, etc.) where the native TEN VAD binaries are not available.

**Key Features:**
- ✅ **ARM64 Compatible**: Works on Linux aarch64 systems
- ✅ **Automatic Fallback**: Native TEN VAD → ONNX TEN VAD → WebRTC VAD
- ✅ **Same Accuracy**: Uses official TEN-VAD ONNX model (Apache 2.0)
- ✅ **Easy Install**: `pipx install "git+https://github.com/tinof/ffsubsync.git#egg=ffsubsync[tenvad-onnx]"`

**Upstream**: This fork is based on [smacke/ffsubsync](https://github.com/smacke/ffsubsync)

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
# or for older versions: sudo yum install ffmpeg
~~~

### Recommended Installation (pipx)

The recommended way to install ffsubsync is using [pipx](https://pypa.github.io/pipx/), which installs the package in an isolated environment and makes the CLI commands globally available.

This fork is the canonical distribution for our builds and includes **ONNX-based TEN VAD** support for ARM64/aarch64 platforms (Oracle Cloud, AWS Graviton, Raspberry Pi, etc.). Native TEN VAD is available on **Linux x64** and **macOS** (Intel and Apple Silicon).

**ARM64/aarch64 (recommended ONNX backend, stable tag):**

~~~
pipx install "ffsubsync[tenvad-onnx] @ git+https://github.com/tinof/ffsubsync@LATEST"
# Older pipx compatibility:
pipx install 'git+https://github.com/tinof/ffsubsync@LATEST#egg=ffsubsync[tenvad-onnx]'
~~~

**Linux x64 / macOS (native TEN VAD - best performance, stable tag):**

~~~
pipx install "ffsubsync[tenvad] @ git+https://github.com/tinof/ffsubsync@LATEST"
# Older pipx compatibility:
pipx install 'git+https://github.com/tinof/ffsubsync@LATEST#egg=ffsubsync[tenvad]'
~~~

**Development head (master):**

~~~
pipx install 'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad-onnx]'
pipx install 'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad]'
~~~

**Minimal installations** (WebRTC VAD fallback):

~~~
pipx install "git+https://github.com/tinof/ffsubsync.git"
~~~

If you already installed without the extra, you can add TEN VAD later:

~~~
# ARM64 systems:
pipx inject ffsubsync onnxruntime

# Linux x64 / macOS:
pipx inject ffsubsync 'ten-vad'
# If PyPI wheels fail to build on your platform, inject from GitHub instead:
pipx inject ffsubsync 'ten-vad @ git+https://github.com/TEN-framework/ten-vad.git'
~~~

If you don't have pipx installed, you can install it first:
~~~
pip install pipx
~~~

### Alternative Installation (pip)

You can also install using pip (requires Python >= 3.9):

**From this fork (stable tag, recommended):**

~~~
# ARM64 / all platforms (ONNX backend)
pip install "ffsubsync[tenvad-onnx] @ git+https://github.com/tinof/ffsubsync@LATEST"
# Compatibility fallback:
pip install 'git+https://github.com/tinof/ffsubsync@LATEST#egg=ffsubsync[tenvad-onnx]'

# Linux x64 / macOS (native TEN VAD)
pip install "ffsubsync[tenvad] @ git+https://github.com/tinof/ffsubsync@LATEST"
# Compatibility fallback:
pip install 'git+https://github.com/tinof/ffsubsync@LATEST#egg=ffsubsync[tenvad]'

# WebRTC only
pip install "ffsubsync @ git+https://github.com/tinof/ffsubsync@LATEST"
~~~

**From the latest development branch via Git (pip or pipx both work):**

~~~
pip install  'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad-onnx]'
pipx install 'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad-onnx]'
pip install  'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad]'
pipx install 'git+https://github.com/tinof/ffsubsync@master#egg=ffsubsync[tenvad]'
~~~

**From a local clone:**

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

**From a local wheel you built:**

~~~
pip install "dist/ffsubsync-<version>-py3-none-any.whl[tenvad-onnx]"  # ONNX backend
pip install "dist/ffsubsync-<version>-py3-none-any.whl[tenvad]"       # native TEN VAD
pip install "dist/ffsubsync-<version>-py3-none-any.whl"                # WebRTC only
~~~

> Note on TEN VAD wheels
>
> - TEN VAD provides Linux x64 Python support and prebuilt libraries; on some
>   platforms a local build may be attempted. If `ten-vad` fails to build, you
>   can either:
>   - install ffsubsync without the extra (defaults to WebRTC VAD), or
>   - inject TEN VAD from GitHub as shown above.
> - On Debian/Ubuntu, you may need: `sudo apt update && sudo apt install libc++1`.

If you choose to install without TEN VAD (no extras), ffsubsync will fall back to the WebRTC VAD automatically.

**VAD Backend Priority:**
1. If `ten-vad` is installed → uses native TEN VAD (best performance)
2. If `onnxruntime` is installed → uses ONNX-based TEN VAD (ARM64 compatible)
3. Otherwise → falls back to WebRTC VAD automatically

### Verifying TEN VAD is active

- Run once with a video; you should see a log like:
  `TEN VAD selected: overriding frame_rate ... 16000`.
- You can also explicitly select it: `--vad=tenvad` or `--vad=subs_then_tenvad`.

**Note:** This tool supports Linux and macOS only. Windows is not supported.

Usage
-----
After installation, three CLI commands are available: `ffs`, `subsync`, and `ffsubsync` (all are equivalent):

Supported subtitle formats: `.srt`, `.ass/.ssa`, MicroDVD `.sub`, and `.vtt`. MicroDVD frame rates are preserved when converting back to `.sub`.

**Basic synchronization with video:**
~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

**Using any of the three command aliases:**
~~~
subsync video.mp4 -i unsynchronized.srt -o synchronized.srt
ffsubsync video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

**Synchronization using reference subtitles:**
There may be occasions where you have a correctly synchronized srt file in a
language you are unfamiliar with, as well as an unsynchronized srt file in your
native language. In this case, you can use the correctly synchronized srt file
directly as a reference for synchronization, instead of using the video as the
reference:

~~~
ffs reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

**Running as a Python module:**
~~~
python -m ffsubsync video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~
The tool uses the file extension to decide whether to perform voice activity
detection on the audio or to directly extract speech from an srt file.

Platform Support
----------------
This tool is designed for command-line usage and supports:
- **Linux** (all major distributions)
- **macOS** (Intel and Apple Silicon)

**Note:** Windows is not supported in this version. The tool is optimized for Unix-like systems.

Requirements
------------
- **Python:** 3.8 or higher
- **ffmpeg:** Must be installed and available in your system PATH
- **Dependencies:** All Python dependencies are automatically installed during package installation

Sync Issues
-----------
If the sync fails, the following recourses are available:
- Try to sync assuming identical video / subtitle framerates by passing
  `--no-fix-framerate`;
- Try passing `--gss` to use [golden-section search](https://en.wikipedia.org/wiki/Golden-section_search)
  to find the optimal ratio between video and subtitle framerates (by default,
  only a few common ratios are evaluated);
- Try a value of `--max-offset-seconds` greater than the default of 60, in the
  event that the subtitles are out of sync by more than 60 seconds (empirically
  unlikely in practice, but possible).
- The default voice activity detector is TEN VAD for low-latency, high-accuracy speech detection. Use `--vad=webrtc` if you prefer the legacy WebRTC backend.

If the sync still fails, consider trying one of the following similar tools:
- [sc0ty/subsync](https://github.com/sc0ty/subsync): does speech-to-text and looks for matching word morphemes
- [kaegi/alass](https://github.com/kaegi/alass): rust-based subtitle synchronizer with a fancy dynamic programming algorithm
- [tympanix/subsync](https://github.com/tympanix/subsync): neural net based approach that optimizes directly for alignment when performing speech detection
- [oseiskar/autosubsync](https://github.com/oseiskar/autosubsync): performs speech detection with bespoke spectrogram + logistic regression
- [pums974/srtsync](https://github.com/pums974/srtsync): similar approach to ffsubsync (WebRTC's VAD + FFT to maximize signal cross correlation)

Speed
-----
`ffsubsync` usually finishes in 20 to 30 seconds, depending on the length of
the video. The most expensive step is actually extraction of raw audio. If you
already have a correctly synchronized "reference" srt file (in which case audio
extraction can be skipped), `ffsubsync` typically runs in less than a second.

VAD Benchmarks
--------------
Performance comparison of different synchronization methods:

| Method | Time | Score | Avg Error |
|--------|------|-------|-----------|
| sub-to-sub | 1.8s | 81,040 | 0.23s |
| audio-webrtc | 30.8s | 62,669 | 0.11s |
| audio-tenvad | 30.5s | 62,669 | 0.11s |

**Notes:** WebRTC and TEN-VAD have identical accuracy. Sub-to-sub (using a
reference subtitle file) is fastest but slightly less accurate than audio-based
synchronization.

How It Works
------------
The synchronization algorithm operates in 3 steps:
1. Discretize both the video file's audio stream and the subtitles into 10ms
   windows.
2. For each 10ms window, determine whether that window contains speech.  This
   is trivial to do for subtitles (we just determine whether any subtitle is
   "on" during each time window); for the audio stream, ffsubsync uses the
   [TEN VAD](https://github.com/TEN-framework/ten-vad) backend by default (if installed), but
   you can switch to WebRTC (`--vad=webrtc`). If TEN VAD is not installed, WebRTC is used.
3. Now we have two binary strings: one for the subtitles, and one for the
   video.  Try to align these strings by matching 0's with 0's and 1's with
   1's. We score these alignments as (# video 1's matched w/ subtitle 1's) - (#
   video 1's matched with subtitle 0's).

The best-scoring alignment from step 3 determines how to offset the subtitles
in time so that they are properly synced with the video. Because the binary
strings are fairly long (millions of digits for video longer than an hour), the
naive O(n^2) strategy for scoring all alignments is unacceptable. Instead, we
use the fact that "scoring all alignments" is a convolution operation and can
be implemented with the Fast Fourier Transform (FFT), bringing the complexity
down to O(n log n).

Limitations
-----------
In most cases, inconsistencies between video and subtitles occur when starting
or ending segments present in video are not present in subtitles, or vice versa.
This can occur, for example, when a TV episode recap in the subtitles was pruned
from video. FFsubsync typically works well in these cases, and in my experience
this covers >95% of use cases. Handling breaks and splits outside of the beginning
and ending segments is left to future work (see below).

Future Work
-----------
Besides general stability and usability improvements, one line
of work aims to extend the synchronization algorithm to handle splits
/ breaks in the middle of video not present in subtitles (or vice versa).
Developing a robust solution will take some time (assuming one is possible).
See [#10](https://github.com/smacke/ffsubsync/issues/10) for more details.

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
- [srt](https://pypi.org/project/srt/) for operating on [SRT files](https://en.wikipedia.org/wiki/SubRip#SubRip_text_file_format)
- [numpy](http://www.numpy.org/) and, indirectly, [FFTPACK](https://www.netlib.org/fftpack/), which powers the FFT-based algorithm for fast scoring of alignments between subtitles (or subtitles and video)
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html), [rich](https://github.com/willmcgugan/rich), and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
Code in this project is [MIT licensed](https://opensource.org/licenses/MIT).
