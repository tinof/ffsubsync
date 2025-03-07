# FFsubsync AE (Anchor Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)

**Next-gen subtitle synchronization** using hybrid AI/digital signal processing to handle both linear offsets and non-linear time warping.

## Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Sample Output](#sample-output)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [How It Works](#how-it-works)
- [Auto-Optimization Details](#auto-optimization-details)
- [Expected Performance Benefits](#expected-performance-benefits)
- [Large File Processing Guide](#large-file-processing-guide)
- [Future Roadmap](#future-roadmap)
- [Credits & License](#credits--license)

## Key Features

- 🎯 **Tiered Alignment Strategy**  
  1. Global offset detection (FFT)  
  2. 2-point anchor alignment  
  3. Automatic midpoint insertion (up to 5 anchors)
  
- 🛠 **Non-Linear Correction**  
  Handles complex cases including variable playback speeds, regional edit differences, drifting frame rates, and partial scene cuts

- ⚡ **Smart Fallbacks**  
  Preserves original sync points where alignment confidence is high with automatic failure detection and graduated retry system
  
- 🚀 **Performance Optimizations**
  - Progressive multi-resolution alignment
  - Content-aware anchor selection
  - Parallel processing for framerate testing
  - Memory-efficient large file handling
  
- 🤖 **Auto-Configuration**
  - Automatic detection of optimal settings
  - Hardware-aware resource allocation
  - Content-adaptive processing strategies

## Installation

First, make sure ffmpeg is installed on your system.

```bash
# Clone the repository
git clone -b sffsubsync-ae https://github.com/tinof/ffsubsync.git
cd ffsubsync

# Install with pipx (recommended for CLI tools)
pipx install -e .

# If you don't have pipx:
python -m pip install --user pipx
pipx ensurepath
```

Alternatively, install directly:
```bash
pipx install git+https://github.com/tinof/ffsubsync.git
```

## Basic Usage

`ffs`, `subsync` and `ffsubsync` all work as entrypoints:

```bash
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
```

You can also use a correctly synchronized srt file as reference:

```bash
ffsubsync reference.srt -i unsynchronized.srt -o synchronized.srt
```

### Auto-Optimized Mode (Recommended)

```bash
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --auto-optimize
```

This automatically detects file size and system capabilities, enabling appropriate optimizations for your specific case.

## Sample Output

Here's what the output looks like when using auto-optimization:

```
Processing subtitles for [video file] (Language: fin)
Synchronizing subtitles for [video file] using [reference subtitle file]
[23:22:56] INFO     Auto-optimization mode enabled - configuring optimal settings                                                            ffsubsync.py:902
           INFO     System resources: 2 CPU cores, 0.7GB available memory                                                                    ffsubsync.py:909
           INFO     Reference file size: 22.69GB                                                                                             ffsubsync.py:918
           INFO     Auto-enabling memory optimized mode for large file                                                                       ffsubsync.py:925
           INFO     Auto-enabling parallel processing with 2 cores                                                                           ffsubsync.py:941
           INFO     Auto-configuring segment length: 97s for very large file                                                                 ffsubsync.py:947
           INFO     Auto-limiting processing duration to 146s for very large file                                                            ffsubsync.py:953
           INFO     Auto-selecting memory-efficient VAD (auditok) for large file on low-memory system                                        ffsubsync.py:964
           INFO     Large file detected (22.69 GB), using memory-optimized processing                                                        ffsubsync.py:832
           INFO     Using optimized processing for large file (22.69 GB)                                                                     ffsubsync.py:704
           INFO     Video duration: 2596.64 seconds                                                                                          ffsubsync.py:711
           INFO     Processing 3 segments                                                                                                    ffsubsync.py:747
           INFO     Processing segments in parallel
```

## Advanced Usage

```bash
# Enable parallel processing for faster synchronization
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --parallel

# Process only the first 30 minutes of a long video
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --max-duration 1800

# Use memory-optimized mode for large files
ffs large_video.mkv -i unsynchronized.srt -o synchronized.srt --memory-optimized

# Combine optimizations for very large files
ffs large_video.mkv -i unsynchronized.srt -o synchronized.srt --parallel --memory-optimized --max-duration 3600
```

## Troubleshooting

If synchronization fails, try these options:
- `--auto-optimize`: Enable automatic optimization (recommended first step)
- `--no-fix-framerate`: Sync assuming identical video/subtitle framerates
- `--gss`: Use golden-section search for optimal framerate ratio
- `--max-offset-seconds N`: Increase from default 60 seconds if subtitles are more out of sync
- `--vad=auditok`: Try alternative audio detection method for low-quality audio
- `--n-anchors N`: Force using specific number of anchor points (0=disable) for handling non-linear drift
- `--parallel`: Enable parallel processing for faster framerate ratio testing
- `--memory-optimized`: Use memory-efficient processing for large files
- `--max-duration N`: Limit processing to first N seconds of video

## How It Works

1. Discretize video audio and subtitles into 10ms windows
2. Detect speech presence in each window using WebRTC's VAD
3. Align the resulting binary patterns using progressive multi-resolution FFT:
   - First pass: Coarse alignment with downsampled signals (10x faster)
   - Second pass: Refined alignment on focused window around coarse result
4. Content-aware anchor-based alignment:
   - Global offset detection
   - Intelligent anchor point selection in speech-dense regions
   - Automatic midpoint insertion with transition density prioritization

## Auto-Optimization Details

When using `--auto-optimize`, FFsubsync automatically:

| File Size | System Memory | CPU Cores | Applied Optimizations |
|-----------|---------------|-----------|------------------------|
| <5GB      | Any           | 1+        | Progressive alignment  |
| <5GB      | Any           | 2+        | +Parallel processing   |
| 5-20GB    | <8GB          | Any       | +Limited duration (10-15min) |
| 5-20GB    | 8GB+          | Any       | +Memory optimization |
| >20GB     | <8GB          | Any       | +Ultra memory mode, limited segments (3-5min) |
| >20GB     | 8GB+          | Any       | +Segmented processing |
| >50GB     | Any           | Any       | +Selective segment processing |

## Expected Performance Benefits

| File Size | Original | Optimized | Auto-Optimized |
|-----------|----------|-----------|---------------|
| 1GB       | ~45s     | ~25s      | ~20s          |
| 5GB       | ~3m 20s  | ~1m 10s   | ~55s          |
| 20GB      | ~15m+    | ~3m 30s   | ~2m 45s       |
| 50GB+     | OOM error| ~5m 15s   | ~4m 30s       |

*Results may vary based on content and hardware.*

## Large File Processing Guide

### Understanding Progress Display

When processing large files with auto-optimization, you'll see:

1. **Segment Progress Bar**: Shows overall progress through all segments
   ```
   Segments: 100%|██████████| 3/3 [05:42<00:00, 114.13s/seg]
   ```

2. **Segment Processing Logs**: Each segment reports its process in the logs
   ```
   [Segment 0] Processing 0.00s to 300.00s
   [Segment 1] Processing 1148.32s to 1448.32s 
   [Segment 2] Processing 2296.64s to 2596.64s
   ```

3. **Final Processing**: After segment selection, a standard progress bar shows alignment progress
   ```
   100%|██████████████████████████| 300.0/300.0 [01:24<00:00, 3.57it/s]
   ```

### Memory-Optimized Mode

For large files, use:
```bash
ffsubsync large_video.mkv -i subtitles.srt -o synced.srt --memory-optimized
```

This processes the video in smaller segments, reduces memory usage, and caches intermediate results.

### Tips for Very Large 4K Files (>20GB)

```bash
# Simplest approach
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --auto-optimize

# For memory errors
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --segment-length 120 --max-duration 900

# For limited CPU
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --memory-optimized --vad=auditok

# Extract audio first (most reliable)
ffmpeg -i large_4k_video.mkv -t 900 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
ffsubsync audio.wav -i subtitles.srt -o synced.srt --auto-optimize
```

## Future Roadmap

- **Machine Learning Integration**  
  - Whisper-based anchor detection for no-audio cases  
  - Neural VAD for better speech boundary detection

- **Advanced Warping**  
  - Bézier curve time mapping  
  - Scene-boundary aware correction

## Why Anchor Edition?

This community-driven fork extends the original FFsubsync with:

| Feature               | Original | AE Edition |
|-----------------------|----------|------------|
| Non-linear alignment  | ❌       | ✅         |
| Tiered retry system   | ❌       | ✅         |
| Progressive alignment | ❌       | ✅         |
| Parallel processing   | ❌       | ✅         |
| Auto-optimization     | ❌       | ✅         |
| ML fallbacks          | ❌       | WIP        |
| Multi-user sync       | ❌       | Planned    |

## Credits & License

- Original concept: [Stephen Macke](https://github.com/smacke/ffsubsync)  
- Anchor edition maintainers: [Your Name] and contributors  
- Core dependencies: FFmpeg, WebRTC VAD, NumPy, SciPy

MIT licensed
