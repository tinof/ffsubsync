#!/usr/bin/env python
"""
TEN-VAD ONNX Backend for ARM64 and platforms without prebuilt binaries.

This module provides an ONNX Runtime-based implementation of TEN-VAD that
replicates the interface of the native ten-vad package.

Copyright © 2025 - ONNX backend implementation for ffsubsync
TEN-VAD model: Copyright © 2025 Agora (Apache License 2.0)
"""

import numpy as np
import onnxruntime as ort
import os
from typing import Tuple
import warnings


def compute_mel_filterbanks(
    fft_bins: np.ndarray,
    n_mels: int = 40,
    sample_rate: int = 16000,
    n_fft: int = 512
) -> np.ndarray:
    """
    Compute mel-filterbank features from FFT magnitude spectrum.

    Args:
        fft_bins: FFT magnitude spectrum [n_fft//2 + 1]
        n_mels: Number of mel-filterbanks (default: 40)
        sample_rate: Audio sampling rate in Hz (default: 16000)
        n_fft: FFT size (default: 512)

    Returns:
        mel_features: Mel-filterbank energies [n_mels]
    """
    # Mel scale conversion functions
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Create mel filterbank
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(8000)  # TEN-VAD uses 0-8000 Hz range

    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Build filterbank matrix
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    # Apply filterbank to FFT bins
    mel_energies = np.dot(fbank, fft_bins ** 2)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

    return np.log(mel_energies)


class TenVadONNX:
    """
    ONNX Runtime-based TEN-VAD implementation.

    This class provides the same interface as the native ten_vad.TenVad class
    but uses ONNX Runtime for inference, enabling support on ARM64 Linux and
    other platforms without prebuilt native libraries.

    Usage:
        vad = TenVadONNX(hop_size=256, threshold=0.5)
        audio_chunk = np.array([...], dtype=np.int16)  # hop_size samples
        probability, is_voice = vad.process(audio_chunk)
    """

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        """
        Initialize TEN-VAD ONNX backend.

        Args:
            hop_size: Number of audio samples per frame (e.g., 256 for 16ms at 16kHz)
            threshold: Voice detection threshold [0.0, 1.0] (default: 0.5)
        """
        self.hop_size = hop_size
        self.threshold = threshold

        # Model constants (from aed_st.h)
        self.n_mels = 40  # AUP_AED_MEL_FILTER_BANK_NUM
        self.n_features = 41  # AUP_AED_FEA_LEN (40 mel + 1 extra)
        self.context_len = 3  # AUP_AED_CONTEXT_WINDOW_LEN
        self.hidden_dim = 64  # AUP_AED_MODEL_HIDDEN_DIM
        self.sample_rate = 16000  # TEN-VAD requires 16kHz

        # FFT configuration
        self.n_fft = 512  # Default FFT size
        self.window = np.hanning(hop_size)

        # Frame buffer for context window (stores last 3 frames)
        self.frame_buffer = np.zeros((self.context_len, self.n_features), dtype=np.float32)
        self.frame_index = 0

        # Hidden states for stateful RNN (4 states, each [1, 64])
        self.hidden_states = [np.zeros((1, self.hidden_dim), dtype=np.float32) for _ in range(4)]

        # Load ONNX model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "onnx_models",
            "ten-vad.onnx"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TEN-VAD ONNX model not found at {model_path}. "
                "Please ensure the model file is included in the package."
            )

        # Create ONNX Runtime session
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.log_severity_level = 3  # ERROR level

            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        if len(self.input_names) != 5 or len(self.output_names) != 5:
            warnings.warn(
                f"Expected 5 inputs and 5 outputs, got {len(self.input_names)} and {len(self.output_names)}. "
                "Model structure may have changed."
            )

    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract mel-filterbank features from audio chunk.

        Args:
            audio_data: Audio samples as int16 numpy array [hop_size]

        Returns:
            features: Feature vector [n_features] (40 mel + 1 extra feature)
        """
        # Convert int16 to float32 and normalize
        audio_float = audio_data.astype(np.float32)

        # Apply window
        if len(audio_float) < self.hop_size:
            # Pad if needed
            padded = np.zeros(self.hop_size, dtype=np.float32)
            padded[:len(audio_float)] = audio_float
            audio_float = padded

        windowed = audio_float[:self.hop_size] * self.window

        # Compute FFT
        fft_result = np.fft.rfft(windowed, n=self.n_fft)
        fft_magnitude = np.abs(fft_result)

        # Compute mel-filterbank features
        mel_features = compute_mel_filterbanks(
            fft_magnitude,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft
        )

        # Compute log energy as the extra feature (41st feature)
        log_energy = np.log(np.sum(audio_float ** 2) + 1e-10)

        # Concatenate: [40 mel + 1 log_energy] = 41 features
        features = np.concatenate([mel_features, [log_energy]])

        return features.astype(np.float32)

    def process(self, audio_data: np.ndarray) -> Tuple[float, int]:
        """
        Process one audio frame for voice activity detection.

        Args:
            audio_data: Audio samples as int16 numpy array of length hop_size

        Returns:
            (probability, is_voice):
                - probability: Voice activity probability [0.0, 1.0]
                - is_voice: Binary voice detection flag (0=no voice, 1=voice)
        """
        # Validate input
        audio_data = np.squeeze(audio_data)
        if audio_data.ndim != 1:
            raise ValueError(f"Audio data must be 1D, got shape {audio_data.shape}")
        if len(audio_data) != self.hop_size:
            raise ValueError(f"Audio data length must be {self.hop_size}, got {len(audio_data)}")
        if audio_data.dtype != np.int16:
            raise ValueError(f"Audio data must be int16, got {audio_data.dtype}")

        # Extract features from current frame
        features = self.extract_features(audio_data)

        # Update frame buffer (shift and append)
        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
        self.frame_buffer[-1] = features

        # Only start inference after we have enough frames
        self.frame_index += 1
        if self.frame_index < self.context_len:
            # Not enough context yet, return neutral
            return 0.0, 0

        # Prepare model inputs
        # Input 0: feature stack [1, 3, 41]
        input_features = self.frame_buffer.reshape(1, self.context_len, self.n_features).astype(np.float32)

        # Inputs 1-4: hidden states [1, 64] each
        inputs = {
            self.input_names[0]: input_features,
            self.input_names[1]: self.hidden_states[0],
            self.input_names[2]: self.hidden_states[1],
            self.input_names[3]: self.hidden_states[2],
            self.input_names[4]: self.hidden_states[3],
        }

        # Run inference
        try:
            outputs = self.session.run(self.output_names, inputs)
        except Exception as e:
            warnings.warn(f"ONNX inference failed: {e}")
            return 0.0, 0

        # Extract probability from output 0 [1, 1, 1]
        probability = float(outputs[0].flatten()[0])
        probability = np.clip(probability, 0.0, 1.0)

        # Update hidden states from outputs 1-4
        for i in range(4):
            self.hidden_states[i] = outputs[i + 1].astype(np.float32)

        # Compute voice flag
        is_voice = 1 if probability >= self.threshold else 0

        return probability, is_voice

    def reset(self):
        """Reset internal state (hidden states and frame buffer)."""
        self.frame_buffer = np.zeros((self.context_len, self.n_features), dtype=np.float32)
        self.frame_index = 0
        self.hidden_states = [np.zeros((1, self.hidden_dim), dtype=np.float32) for _ in range(4)]

    def __del__(self):
        """Cleanup ONNX session."""
        if hasattr(self, 'session'):
            del self.session


# Alias for compatibility with native ten_vad package
TenVad = TenVadONNX
