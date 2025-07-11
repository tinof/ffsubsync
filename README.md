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

The recommended way to install ffsubsync is using [pipx](https://pypa.github.io/pipx/), which installs the package in an isolated environment and makes the CLI commands globally available:

~~~
pipx install ffsubsync
~~~

If you don't have pipx installed, you can install it first:
~~~
pip install pipx
~~~

### Alternative Installation (pip)

You can also install using pip (requires Python >= 3.8):
~~~
pip install git+https://github.com/tinof/ffsubsync@latest
~~~

For the latest development version:
~~~
pipx install git+https://github.com/tinof/ffsubsync@latest
~~~

**Note:** This tool supports Linux and macOS only. Windows is not supported.

Usage
-----
After installation, three CLI commands are available: `ffs`, `subsync`, and `ffsubsync` (all are equivalent):

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
- Try `--vad=auditok` since [auditok](https://github.com/amsehili/auditok) can
  sometimes work better in the case of low-quality audio than WebRTC's VAD.
  Auditok does not specifically detect voice, but instead detects all audio;
  this property can yield suboptimal syncing behavior when a proper VAD can
  work well, but can be effective in some cases.

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

How It Works
------------
The synchronization algorithm operates in 3 steps:
1. Discretize both the video file's audio stream and the subtitles into 10ms
   windows.
2. For each 10ms window, determine whether that window contains speech.  This
   is trivial to do for subtitles (we just determine whether any subtitle is
   "on" during each time window); for the audio stream, use an off-the-shelf
   voice activity detector (VAD) like
   the one built into [webrtc](https://webrtc.org/).
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
