"""Measure four signal properties on the same mono PCM Whisper receives."""

from pathlib import Path

import numpy as np


AUDIO_FEATURE_NAMES = ("audio_rms", "audio_near_silent_fraction", "audio_full_scale_fraction", "audio_frame_rms_std")
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms, nonoverlapping.


def measure_waveform(waveform) -> dict[str, float]:
    """Measure float PCM at 16 kHz on the unit-amplitude scale.

    Per-frame RMS includes a final short frame. The near-silent threshold is
    0.01 RMS (-40 dBFS). Samples at or above 0.999 absolute amplitude are
    counted as near full scale. That is a proxy for clipping in the decoded
    waveform, since resampling or downmixing can hide clipping in the source.
    """
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("Expected one mono audio channel.")
    if not samples.size or not np.isfinite(samples).all():
        return dict.fromkeys(AUDIO_FEATURE_NAMES, float("nan"))
    starts = np.arange(0, samples.size, FRAME_SAMPLES)
    lengths = np.minimum(samples.size - starts, FRAME_SAMPLES)
    squared = np.square(samples, dtype=np.float64)
    frame_rms = np.sqrt(np.add.reduceat(squared, starts) / lengths)
    return {
        "audio_rms": float(np.sqrt(squared.mean())),
        "audio_near_silent_fraction": float(np.mean(frame_rms <= .01)),
        "audio_full_scale_fraction": float(np.mean(np.abs(samples) >= .999)),
        "audio_frame_rms_std": float(frame_rms.std()),
    }


def extract_audio_features(audio_path: str | Path) -> dict[str, float]:
    """Decode an upload with faster-whisper and measure it without text inputs."""
    import av
    from faster_whisper.audio import decode_audio

    try:
        waveform = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    except (av.error.FFmpegError, OSError, ValueError):
        return dict.fromkeys(AUDIO_FEATURE_NAMES, float("nan"))
    return measure_waveform(waveform)
