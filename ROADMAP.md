# FFsubsync Fork Roadmap

This document serves as a development handoff for continuing work on this fork of ffsubsync.

## Current State (January 2025)

### Fork Features vs Upstream

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Basic sync (offset + scale) | ✅ | ✅ |
| WebRTC VAD | ✅ | ✅ |
| TEN VAD (native) | ✅ | ✅ |
| TEN VAD (ONNX/ARM64) | ❌ | ✅ |
| Piecewise sync (mid-file drift) | ❌ | ✅ |

### Repository

- **Upstream**: https://github.com/smacke/ffsubsync
- **This fork**: https://github.com/tinof/ffsubsync

---

## Completed Features

### 1. ONNX-based TEN VAD for ARM64

**Problem**: Native TEN VAD binaries only available for x86_64. ARM64 users (Oracle Cloud, AWS Graviton, Raspberry Pi) couldn't use high-accuracy VAD.

**Solution**: Added ONNX Runtime backend that loads official TEN-VAD ONNX model.

**Files**:
- `ffsubsync/speech_transformers.py` - ONNX detector implementation
- `pyproject.toml` - `tenvad-onnx` extra

**Usage**:
```bash
pip install ffsubsync[tenvad-onnx]
ffs video.mkv -i input.srt -o output.srt --vad=tenvad
```

### 2. Piecewise Sync for Mid-File Drift

**Problem**: Standard ffsubsync applies single global offset+scale. Fails when:
- Subtitles drift mid-file (different video edits)
- Commercial breaks removed differently between releases
- Translations have different segmentation

**Solution**: Multi-phase algorithm:
1. **Ratio-based time warping** - Maps source timeline proportionally to reference
2. **Local offset correction** - Finds optimal offset per 60-second window using median filtering
3. **Smooth interpolation** - Blends corrections to avoid jarring jumps

**Files**:
- `ffsubsync/tools/piecewise_sync.py` - Standalone tool
- `README.md` - Documentation

**Real-world result**: Improved sync from ~60% to 98%+ within 1 second of reference.

**Usage**:
```bash
python -m ffsubsync.tools.piecewise_sync reference.srt input.srt output.srt --verify
```

---

## Roadmap: Future Work

### Priority 1: Integration into Main CLI

**Goal**: Add `--piecewise` flag to main `ffs` command.

**Current state**: Piecewise sync is a separate tool. Users must call it manually.

**Proposed implementation**:
```bash
ffs reference.srt -i input.srt -o output.srt --piecewise
ffs video.mkv -i input.srt -o output.srt --piecewise  # Extract embedded sub as reference
```

**Technical approach**:
1. Import `piecewise_sync` function from `ffsubsync.tools.piecewise_sync`
2. Add `--piecewise` argparse flag in `ffsubsync.py`
3. When flag is set AND reference is subtitle (not audio):
   - Use piecewise sync instead of standard FFT alignment
4. When flag is set AND reference is video:
   - First extract embedded subtitle (if available)
   - Fall back to standard sync if no embedded sub

**Estimated effort**: 2-3 hours

### Priority 2: Automatic Piecewise Detection

**Goal**: Automatically detect when piecewise sync is needed.

**Proposed heuristics**:
1. Run standard sync first
2. Verify sync quality (% within threshold)
3. If quality < 90%, automatically try piecewise
4. Report which method was used

**Technical approach**:
1. Add `verify_sync()` call after standard sync
2. If metrics below threshold, re-run with piecewise
3. Add `--auto` flag to enable this behavior

**Estimated effort**: 3-4 hours

### Priority 3: Scene-Break Anchor Detection

**Goal**: Use detected scene breaks as anchor points for more precise alignment.

**Current limitation**: Piecewise sync uses fixed 60-second windows. Scene breaks may not align with window boundaries.

**Proposed improvement**:
1. Detect gaps > N seconds in both subtitle streams
2. Match gaps by temporal proximity
3. Use matched gaps as anchor points
4. Apply piecewise interpolation between anchors

**Already partially implemented** in development (gap detection code exists in session history).

**Estimated effort**: 4-6 hours

### Priority 4: Console Entry Point

**Goal**: Add `piecewise-sync` as installable command.

**Implementation**:
Add to `pyproject.toml`:
```toml
[project.scripts]
piecewise-sync = "ffsubsync.tools.piecewise_sync:main"
```

**Estimated effort**: 15 minutes

### Priority 5: Subtitle Merger Enhancement

**Goal**: When translations have different segmentation (different number of lines), intelligently merge/split.

**Problem observed**: Finnish translation had 486 subs, French had 587. Some gaps are unfixable by timing alone - translator combined/split lines differently.

**Possible approaches**:
1. Content-aware matching using subtitle duration similarity
2. Speech-to-text comparison (expensive but accurate)
3. Accept "best effort" and document limitation

**Estimated effort**: Research needed, potentially 1-2 days

---

## Technical Notes

### Key Files

| File | Purpose |
|------|---------|
| `ffsubsync/ffsubsync.py` | Main entry point, CLI parsing |
| `ffsubsync/aligners.py` | FFT-based alignment algorithms |
| `ffsubsync/speech_transformers.py` | VAD backends (WebRTC, TEN, ONNX) |
| `ffsubsync/subtitle_transformers.py` | Subtitle manipulation (shift, scale, merge) |
| `ffsubsync/tools/piecewise_sync.py` | Piecewise sync tool |

### Algorithm Details

**Standard sync** (FFTAligner):
1. Convert both streams to binary (speech=1, silence=0) at 10ms resolution
2. FFT convolution to find best alignment
3. Single offset + scale factor

**Piecewise sync**:
1. Ratio warp: `new_time = ref_start + (old_time - input_start) * (ref_duration / input_duration)`
2. Local correction: For each 60s window, find median offset between matched subtitles
3. Interpolation: Blend corrections between windows to avoid jumps

### External Tools Integration

The `ssync` script (`/home/ubuntu/bin/ssync`) wraps ffsubsync for batch processing:
- Finds `.fi.srt` files matching video files
- Extracts embedded subtitles from MKV as reference
- Supports `-p` flag for piecewise sync

---

## Testing Notes

### Test Case: I, Jack Wright (2025) S01E01

**Scenario**: Finnish subtitles from OpenSubtitles, French embedded in MKV.

**Problem**: ~60% sync accuracy with standard method. Subtitles drifted mid-file.

**Root cause**: Different video edits - Finnish subs were for different release.

**Solution**: Piecewise sync achieved 98%+ accuracy.

**Remaining issues** (4 segments with 2-4s gaps):
- Translation segmentation differences (unfixable by timing)
- On-screen text annotations in French not present in Finnish

### Character Encoding

**Issue encountered**: Finnish characters (ä, ö) corrupted during processing.

**Solution**: 
1. Detect encoding with `chardet`
2. Write output with UTF-8 BOM (`utf-8-sig`) for TV compatibility

---

## Contributing

When continuing development:

1. Run tests: `pytest -v -m 'not integration' tests/`
2. Run linter: `ruff check ffsubsync/ --fix`
3. Run formatter: `ruff format ffsubsync/`
4. Test piecewise sync: `python -m ffsubsync.tools.piecewise_sync --help`

---

## Session History Context

This fork was developed across multiple sessions. Key decisions:

1. **Why ONNX for ARM64?**: Native TEN VAD requires specific binaries. ONNX Runtime is cross-platform.

2. **Why piecewise sync as separate tool?**: 
   - Minimize changes to core ffsubsync code
   - Can be tested independently
   - Future integration planned (see Priority 1)

3. **Why 60-second windows?**: 
   - Balance between granularity and stability
   - Matches typical scene length
   - Configurable via `--window` flag

4. **Why median filtering for offsets?**:
   - Robust to outliers (mismatched subtitles)
   - Better than mean for non-normal distributions
