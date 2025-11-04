# TEN-VAD ONNX Model

This directory contains the ONNX model file for TEN-VAD (Voice Activity Detection).

## Model Information

- **Source**: [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad)
- **License**: Apache License 2.0
- **Model Version**: 1.0
- **Model File**: `ten-vad.onnx` (309 KB)

## Model Architecture

- **Input 0**: Audio features `[1, 3, 41]` - 3 consecutive frames of 41-dimensional mel-filterbank features
  - 40 mel-filterbanks (0-8000 Hz range)
  - 1 log-energy feature
- **Inputs 1-4**: Hidden states `[1, 64]` each - Stateful RNN (LSTM/GRU) internal states
- **Output 0**: Voice probability `[1, 1, 1]` - Speech detection probability [0.0, 1.0]
- **Outputs 1-4**: Updated hidden states `[1, 64]` each - To be fed back into next inference

## Requirements

- **Audio Format**: 16-bit PCM, mono, 16 kHz sampling rate
- **Frame Size**: Configurable hop_size (default: 256 samples = 16ms at 16kHz)
- **Runtime**: ONNX Runtime 1.17.1 or higher

## Attribution

TEN VAD Model:
```
Copyright © 2025 Agora
Licensed under the Apache License, Version 2.0
```

From the TEN-VAD project:
> "We selected TEN VAD because it provides faster and more accurate sentence-end
> detection in Japanese compared to other VADs, while still being lightweight
> and fast enough for live use." - LiveCap, Hakase shojo

> "TEN VAD's overall performance is better than Silero VAD. Its high accuracy
> and low resource consumption helped us improve efficiency and significantly
> reduce costs." - Rustpbx

## Why ONNX?

The ONNX version of TEN-VAD enables deployment on platforms without prebuilt native libraries,
specifically:
- **Linux ARM64 (aarch64)** - Oracle Cloud, AWS Graviton, etc.
- Any platform supported by ONNX Runtime

The native `ten-vad` Python package only supports:
- Linux x64
- macOS (Intel and Apple Silicon)
- Windows x64/x86

For platforms with native support, the original `ten-vad` package is recommended for optimal performance.

## References

- TEN-VAD GitHub: https://github.com/TEN-framework/ten-vad
- TEN-VAD Model Release: June 2025
- ONNX Runtime: https://onnxruntime.ai/
