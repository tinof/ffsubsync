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


Language-agnostic automatic synchronization of subtitles with video, so that
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
**Recommended Method (using pipx):**

[pipx](https://github.com/pypa/pipx) installs Python applications into isolated environments, making them available globally without interfering with system packages.

1.  **Install ffmpeg:** `ffsubsync` requires `ffmpeg` to process video files.
    *   On macOS: `brew install ffmpeg`
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    *   On other systems: Follow the official [ffmpeg installation guide](https://ffmpeg.org/download.html).

2.  **Install pipx:** If you don't have `pipx` installed, follow the [official pipx installation guide](https://pipx.pypa.io/stable/installation/).

3.  **Install ffsubsync:**
    ~~~
    pipx install ffsubsync
    ~~~

**Alternative Method (using pip):**

If you prefer to manage the environment yourself or are developing `ffsubsync`:

1.  **Install ffmpeg:** (See step 1 above)

2.  **Install ffsubsync:**
    ~~~
    pip install ffsubsync
    ~~~

**Installing the Latest Development Version:**

To install the latest code directly from GitHub (use with caution):
~~~
pipx install git+https://github.com/smacke/ffsubsync.git
# Or using pip:
# pip install git+https://github.com/smacke/ffsubsync.git
~~~

Usage
-----
`ffs`, `subsync` and `ffsubsync` all work as entrypoints:
~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

There may be occasions where you have a correctly synchronized srt file in a
language you are unfamiliar with, as well as an unsynchronized srt file in your
native language. In this case, you can use the correctly synchronized srt file
directly as a reference for synchronization, instead of using the video as the
reference:

~~~
ffsubsync reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

`ffsubsync` uses the file extension to decide whether to perform voice activity
detection on the audio or to directly extract speech from an srt file.

Alignment Modes & Sync Issues
-----------------------------

`ffsubsync` now features two main alignment strategies:

1.  **Standard Mode (Default):** This mode finds the single best global time offset and framerate scaling factor to align the subtitles. It works well for most common cases where the desynchronization is consistent throughout the file.
2.  **Anchor-Based Mode (Experimental):** This mode is designed to handle more complex synchronization issues, such as non-linear drift (where the timing difference changes over time) or structural differences caused by edits (like commercial breaks). It works by:
    *   Performing an initial global alignment.
    *   Iteratively finding multiple high-confidence "anchor points" throughout the media.
    *   Applying piecewise time adjustments ("warping") between these anchor points.

**Automatic Fallback (Default Behavior):**

By default, `ffsubsync` uses an intelligent tiered approach:
*   It first attempts synchronization using the **Standard Mode**.
*   It evaluates the quality of the standard alignment based on the correlation score and the calculated offset.
*   If the standard alignment appears poor (low score or very large offset, based on internal thresholds), it automatically attempts a fallback using the **Anchor-Based Mode**.
*   The result from the anchor-based fallback is used only if its internal confidence score is sufficient; otherwise, the tool reverts to the standard result (with warnings).

This default behavior aims to provide the robustness of the anchor-based method when needed, without sacrificing the speed and simplicity of the standard method for common cases.

**Manual Control & Troubleshooting:**

If synchronization fails or produces suboptimal results with the default settings:

*   **Force Anchor Mode:** You can explicitly force the use of the anchor-based aligner by adding the `--use-anchor-based-aligner` (or `--anchor-mode`) flag. This can be helpful if you suspect non-linear drift that the automatic detection might have missed.
    ~~~
    ffs video.mp4 -i unsync.srt -o sync.srt --use-anchor-based-aligner
    ~~~
*   **Adjust Standard Mode Parameters:**
    *   Try `--no-fix-framerate` if you suspect the framerate is actually correct.
    *   Try `--gss` to perform a more thorough search for the optimal framerate ratio.
    *   Increase `--max-offset-seconds` if the offset might be larger than 60 seconds.
*   **Try a Different VAD:** Use `--vad=auditok` or `--vad=silero` (if installed) if the default VAD struggles with the audio quality. `auditok` detects general audio activity, while `silero` is another speech-specific VAD.

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
   1's. We score these alignments based on maximizing the cross-correlation between the signals.

The standard alignment mode finds the single global offset that maximizes this correlation across the entire signal using the Fast Fourier Transform (FFT) for efficiency (O(n log n)).

The anchor-based mode extends this by first finding a global offset, then iteratively identifying high-confidence local alignment points (anchors) within segments, and finally warping the subtitle timeline based on these anchors.

Limitations
-----------
While the new anchor-based mode aims to handle more complex synchronization issues like non-linear drift and mid-file breaks/edits (e.g., commercial insertions/deletions), its effectiveness can depend on the quality of the VAD signal and the distinctiveness of speech patterns around potential anchor points.

*   **Threshold Tuning:** The automatic fallback logic relies on internal thresholds for score and offset, which are empirically determined and might need adjustment for certain types of content.
*   **Anchor Confidence:** The confidence calculation for anchors is based on local signal correlation. Very noisy audio or long periods of silence/music can still make it difficult to find reliable anchors.
*   **Extreme Edits:** Very large or numerous insertions/deletions might still pose a challenge, although the anchor mode is better equipped to handle them than the standard mode.

Future Work
-----------
*   Refining the anchor selection and confidence scoring logic.
*   Implementing segment-level fallbacks within the anchor mode (e.g., reverting specific segments to linear interpolation if local confidence is too low).
*   Improving the automatic detection thresholds for the tiered fallback mechanism.
*   Exploring alternative local alignment algorithms (e.g., Dynamic Time Warping) as potential additions or replacements for local FFT within the anchor mode.
*   General stability and usability improvements.

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
