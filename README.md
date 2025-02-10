FFsubsync AE (Anchor Edition)
=============================

[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)

**Next-gen subtitle synchronization** using hybrid AI/digital signal processing to handle both linear offsets and non-linear time warping.

Key Features
------------
- 🎯 **Tiered Alignment Strategy**  
  1. Global offset detection (FFT)  
  2. 2-point anchor alignment  
  3. Automatic midpoint insertion (up to 5 anchors)
  
- 🛠 **Non-Linear Correction**  
  Handles complex cases:  
  - Variable playback speeds  
  - Regional edit differences  
  - Drifting frame rates  
  - Partial scene cuts

- ⚡ **Smart Fallbacks**  
  - Preserves original sync points where alignment confidence is high  
  - Automatic failure detection with graduated retry system

Install
-------
First, make sure ffmpeg is installed on your system.

For the fastest installation experience:
~~~
# Clone the repository
git clone -b sffsubsync-ae https://github.com/tinof/ffsubsync.git
cd ffsubsync

# Install with pipx (recommended for CLI tools)
pipx install -e .

# If you don't have pipx:
python -m pip install --user pipx
pipx ensurepath
~~~

Alternatively, you can install directly:
~~~
pipx install git+https://github.com/tinof/ffsubsync.git@ffsubsync-ae
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
- `--n-anchors N`: Force using specific number of anchor points (0=disable) for handling non-linear drift

How It Works
------------
1. Discretize video audio and subtitles into 10ms windows
2. Detect speech presence in each window using WebRTC's VAD
3. Align the resulting binary patterns using FFT-based convolution
4. Anchor-based alignment:
   - Global offset detection
   - 2-point anchor alignment
   - Automatic midpoint insertion

Future Roadmap
--------------
- **Machine Learning Integration**  
  - Whisper-based anchor detection for no-audio cases  
  - Neural VAD for better speech boundary detection

- **Cloud-Native Features**  
  - Distributed alignment using serverless FFT  
  - Crowdsourced synchronization patterns

- **Advanced Warping**  
  - Bézier curve time mapping  
  - Scene-boundary aware correction

- **Real-Time Collaboration**  
  - Shared anchor editing via WebRTC data channels  
  - Version-controlled subtitle histories

Why Anchor Edition?
-------------------
This community-driven fork extends the original FFsubsync with:

| Feature               | Original | AE Edition |
|-----------------------|----------|------------|
| Non-linear alignment  | ❌       | ✅         |
| Tiered retry system   | ❌       | ✅         |
| ML fallbacks          | ❌       | WIP        |
| Multi-user sync       | ❌       | Planned    |

Credits
-------
- Original concept: [Stephen Macke](https://github.com/smacke/ffsubsync)  
- Anchor edition maintainers: [Your Name] and contributors  
- Core dependencies: FFmpeg, WebRTC VAD, NumPy, SciPy

*"Time is an illusion. Subtitle synchronization doubly so."* - Community Proverb

License
-------
MIT licensed
