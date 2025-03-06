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
  
- 🚀 **Performance Optimizations**
  - Progressive multi-resolution alignment
  - Content-aware anchor selection
  - Parallel processing for framerate testing
  - Memory-efficient large file handling
  
- 🤖 **Auto-Configuration**
  - Automatic detection of optimal settings
  - Hardware-aware resource allocation
  - Content-adaptive processing strategies
  - Self-tunes for different file sizes

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
pipx install git+https://github.com/tinof/ffsubsync.git
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

### Auto-Optimized Mode (Recommended)

For the best experience with any file type, use auto-optimization:

~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --auto-optimize
~~~

This will:
- Automatically detect file size and system capabilities
- Enable appropriate optimizations for your specific case
- Adjust resource allocation based on available CPU and memory
- Configure optimal processing strategy for content type

Advanced Usage
-------------
~~~
# Enable parallel processing for faster synchronization
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --parallel

# Process only the first 30 minutes of a long video
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt --max-duration 1800

# Use memory-optimized mode for large files
ffs large_video.mkv -i unsynchronized.srt -o synchronized.srt --memory-optimized

# Combine optimizations for very large files
ffs large_video.mkv -i unsynchronized.srt -o synchronized.srt --parallel --memory-optimized --max-duration 3600
~~~

Sync Troubleshooting
-----------
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

How It Works
------------
1. Discretize video audio and subtitles into 10ms windows
2. Detect speech presence in each window using WebRTC's VAD
3. Align the resulting binary patterns using progressive multi-resolution FFT:
   - First pass: Coarse alignment with downsampled signals (10x faster)
   - Second pass: Refined alignment on focused window around coarse result
4. Content-aware anchor-based alignment:
   - Global offset detection
   - Intelligent anchor point selection in speech-dense regions
   - Automatic midpoint insertion with transition density prioritization

Auto-Optimization Details
------------------------
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

The system automatically adapts to your hardware capabilities and file characteristics, making intelligent trade-offs to ensure successful synchronization with optimal performance.

Performance Benchmarks
---------------------
| File Size | Original | Optimized | Auto-Optimized |
|-----------|----------|-----------|---------------|
| 1GB       | 45s      | 25s       | 20s           |
| 5GB       | 3m 20s   | 1m 10s    | 55s           |
| 20GB      | 15m+     | 3m 30s    | 2m 45s        |
| 50GB+     | OOM error| 5m 15s    | 4m 30s        |

*Tested on 8-core system with 16GB RAM. Results may vary based on content and hardware.*

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
| Progressive alignment | ❌       | ✅         |
| Parallel processing   | ❌       | ✅         |
| Auto-optimization     | ❌       | ✅         |
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

## Processing Large Video Files

When working with very large video files (>10GB), you may encounter memory or performance issues. Here are some tips for optimizing the synchronization process:

### Recommended Approach: Auto-Optimization

The simplest solution for large files is to use auto-optimization:

```bash
ffsubsync large_video.mkv -i subtitles.srt -o synced.srt --auto-optimize
```

This automatically:
- Detects your file size and system capabilities
- Enables appropriate optimizations (memory, progressive, parallel)
- Configures optimal segment sizes and processing duration
- Selects the best VAD for your hardware constraints

### Use Memory-Optimized Mode

This mode:
- Processes the video in smaller segments
- Reduces memory usage
- Caches intermediate results
- May be slower but more reliable for large files

```bash
ffsubsync large_video.mkv -i subtitles.srt -o synced.srt --memory-optimized
```

### Automatic Size-Based Optimization

FFSubSync AE now automatically adapts its processing strategy based on file size:

- **1-5GB files**: Standard processing with optimized memory usage
- **5-20GB files**: Segment-based processing with checkpointing
- **20-80GB files**: Selective processing of representative portions

### Progressive Multi-Resolution Alignment

For faster processing of any file size:

```bash
ffsubsync video.mkv -i subtitles.srt -o synced.srt --progressive
```

This approach:
- First performs a coarse alignment on downsampled signals (5-10x faster)
- Then refines the alignment on a focused window around the coarse result
- Maintains accuracy while significantly reducing processing time

### Parallel Processing

Enable parallel processing to test multiple framerate ratios simultaneously:

```bash
ffsubsync video.mkv -i subtitles.srt -o synced.srt --parallel
```

Requirements:
- Multicore CPU (benefits scale with available cores)
- At least 4GB RAM (8GB+ recommended for large files)

### Content-Aware Anchor Selection

Intelligently selects anchor points based on speech pattern complexity:

```bash
ffsubsync video.mkv -i subtitles.srt -o synced.srt --content-aware
```

This feature:
- Identifies regions with high speech transition density
- Prioritizes distinctive speech patterns for anchor placement
- Improves non-linear correction accuracy

### Other Optimization Tips

1. **Extract Audio First**: For very large files, extract the audio track first and use it as reference:
   ```
   ffmpeg -i large_video.mkv -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
   ffsubsync audio.wav -i subtitles.srt -o synced.srt
   ```

2. **Use Specific Audio Stream**: If your video has multiple audio tracks, specify which one to use:
   ```
   ffsubsync --reference-stream 0:a:0 large_video.mkv -i subtitles.srt -o synced.srt
   ```

3. **Limit Processing Duration**: Process only a portion of the video:
   ```
   ffsubsync --start-seconds 0 --max-duration 600 large_video.mkv -i subtitles.srt -o synced.srt
   ```

4. **Clean Cache**: Remove cached files if you encounter issues:
   ```
   rm -rf /path/to/video/directory/.ffsubsync_cache
   ```

5. **Low-Resource Systems**: For systems with limited RAM (<4GB):
   ```
   ffsubsync --memory-optimized --vad=auditok --no-fix-framerate large_video.mkv -i subtitles.srt -o synced.srt
   ```

6. **4K Video Processing**: For high-resolution videos:
   ```
   ffsubsync --auto-optimize 4k_video.mkv -i subtitles.srt -o synced.srt
   ```

### Technical Details

The optimizations include:
- Progressive multi-resolution FFT alignment (5-10x faster)
- Content-aware anchor selection for better non-linear correction
- Parallel processing of framerate ratios (scales with CPU cores)
- Segment-based processing to reduce memory footprint
- Disk caching of intermediate results
- Optimized FFmpeg commands with thread control
- Automatic resource limiting to prevent crashes
- Checkpoint system for resumable processing

## Processing Very Large 4K Files

For extremely large 4K files (>20GB), the simplest approach is to use auto-optimization:

```bash
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --auto-optimize
```

If you encounter issues with auto-optimization on very large files, try these specific solutions:

1. **Memory Errors**

If you see "out of memory" errors:
```bash
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --segment-length 120 --max-duration 900
```
This uses shorter segments (2 minutes) and limits processing to the first 15 minutes.

2. **Processor-Related Errors**

On systems with limited CPU:
```bash
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --memory-optimized --vad=auditok
```
This uses a less CPU-intensive voice detection algorithm.

3. **Extract Audio First**

For most reliable processing of very large 4K files:
```bash
# Extract first 15 minutes of audio
ffmpeg -i large_4k_video.mkv -t 900 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav

# Use the extracted audio for synchronization
ffsubsync audio.wav -i subtitles.srt -o synced.srt --auto-optimize
```

4. **Last Resort for Extremely Limited Systems**

On systems with very limited resources:
```bash
ffsubsync large_4k_video.mkv -i subtitles.srt -o synced.srt --segment-length 60 --max-duration 600 --vad=auditok --no-fix-framerate --memory-optimized
```
