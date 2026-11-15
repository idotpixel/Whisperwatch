"""Live English transcription with the frozen faster-whisper Large V3 settings."""

from functools import lru_cache
from math import isfinite
from os import environ
from pathlib import Path

from .features import extract_features
from .inference import predict_from_features


WHISPER_REPO = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
MAX_DURATION_S = 30


@lru_cache(maxsize=1)
def _whisper_model(device: str, compute_type: str):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION, allow_patterns=["*.json", "model.bin"])
    return WhisperModel(model_path, device=device, compute_type=compute_type)


def _asr_settings() -> tuple[str, str]:
    device = environ.get("WHISPERWATCH_ASR_DEVICE", "cpu")
    allowed = {"cuda": ("float16",), "cpu": ("float32", "int8")}
    if device not in allowed:
        raise ValueError("WHISPERWATCH_ASR_DEVICE must be cpu or cuda.")
    compute_type = environ.get("WHISPERWATCH_ASR_COMPUTE_TYPE", allowed[device][0])
    if compute_type not in allowed[device]:
        raise ValueError(f"Choose an ASR compute type for {device}: {', '.join(allowed[device])}.")
    return device, compute_type


def transcribe_audio(audio_path: str) -> dict:
    """Return text, faster-whisper segment objects, and source duration in seconds."""
    return _transcribe_audio(audio_path)


def _transcribe_audio(audio_path: str, max_duration_s: int | None = None) -> dict:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.stat().st_size:
        raise ValueError("The audio file is empty.")
    segments, info = _whisper_model(*_asr_settings()).transcribe(
        str(path), language="en", task="transcribe", beam_size=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], vad_filter=False, word_timestamps=False,
    )
    if not isfinite(info.duration) or info.duration <= 0:
        raise ValueError("The file has no audio of positive duration.")
    if max_duration_s is not None and info.duration > max_duration_s:
        raise ValueError(f"WhisperWatch scoring supports English clips of up to {max_duration_s} seconds.")
    segments = list(segments)
    return {
        "transcript": "".join(segment.text for segment in segments).strip(),
        "segments": segments,
        "duration_s": info.duration,
    }


def analyze_audio(audio_path: str) -> dict:
    """Transcribe English speech and score it with the archived-feature model."""
    audio = _transcribe_audio(audio_path, max_duration_s=MAX_DURATION_S)
    features = extract_features(audio["segments"], audio["transcript"], audio["duration_s"])
    result = predict_from_features(features)
    result["details"]["asr_device"], result["details"]["asr_compute_type"] = _asr_settings()
    result["details"]["asr_compatibility"] = "experimental: archived decoding settings are incomplete"
    return {"transcript": audio["transcript"], **result}
