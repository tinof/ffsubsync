FFsubsync
=======

[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)

Language-agnostic automatic synchronization of subtitles with video, so that
subtitles are aligned to the correct starting point within the video.

This is a CLI-focused fork of the original [ffsubsync](https://github.com/smacke/ffsubsync) by Stephen Macke.

Install
-------
First, make sure ffmpeg is installed. On MacOS:
~~~
brew install ffmpeg
~~~
(Windows users: ensure `ffmpeg` is on your path)

For the fastest installation experience:
~~~
# Clone the repository
git clone -b sliding-context https://github.com/tinof/ffsubsync.git
cd ffsubsync

# Install with pipx (recommended for CLI tools)
pipx install -e .

# If you don't have pipx:
python -m pip install --user pipx
pipx ensurepath
~~~

Alternatively, you can install directly (slower as it needs to clone the repository):
~~~
pipx install git+https://github.com/tinof/ffsubsync.git@sliding-context
~~~

Usage
-----
`ffs`, `subsync` and `ffsubsync` all work as entrypoints:
~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

You can also use a correctly synchronized srt file as reference instead of video:
~~~
ffsubsync reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

Sync Troubleshooting
-----------
If synchronization fails, try these options:
- `--no-fix-framerate`: Sync assuming identical video/subtitle framerates
- `--gss`: Use golden-section search for optimal framerate ratio
- `--max-offset-seconds N`: Increase from default 60 seconds if subtitles are more out of sync
- `--vad=auditok`: Try alternative audio detection method for low-quality audio

How It Works
------------
1. Discretize video audio and subtitles into 10ms windows
2. Detect speech presence in each window using WebRTC's VAD
3. Align the resulting binary patterns using FFT-based convolution

Credits
-------
- Original project by [Stephen Macke](https://github.com/smacke/ffsubsync)
- Core dependencies: ffmpeg, webrtc VAD, numpy
- Full credits and documentation available in the [original repository](https://github.com/smacke/ffsubsync)

License
-------
MIT licensed
