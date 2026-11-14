"""Optional live English transcription with faster-whisper Large V3."""

from functools import lru_cache
from os import environ
from pathlib import Path

from .features import extract_features
from .inference import predict_from_features


@lru_cache(maxsize=1)
def _whisper_model(device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel("large-v3", device=device, compute_type=compute_type)


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
    """Return text, faster-whisper segments, and source duration in seconds.

    This experimental path uses faster-whisper's standard decoding options.
    Whispered does not publish enough options to establish exact decode parity.
    Set WHISPERWATCH_ASR_DEVICE=cuda for GPU float16, the archived precision.
    On CPU, WHISPERWATCH_ASR_COMPUTE_TYPE=int8 explicitly enables quantization.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    segments, info = _whisper_model(*_asr_settings()).transcribe(str(path), language="en", task="transcribe")
    segments = list(segments)
    return {
        "transcript": "".join(segment.text for segment in segments).strip(),
        "segments": segments,
        "duration_s": info.duration,
    }


def analyze_audio(audio_path: str) -> dict:
    """Transcribe English speech and score it with the archived-feature model."""
    audio = transcribe_audio(audio_path)
    if audio["duration_s"] > 30:
        raise ValueError("WhisperWatch scoring supports English clips of up to 30 seconds.")
    features = extract_features(audio["segments"], audio["transcript"], audio["duration_s"])
    result = predict_from_features(features)
    result["details"]["asr_compute_type"] = _asr_settings()[1]
    result["details"]["asr_compatibility"] = "experimental: archived decoding settings are incomplete"
    return {"transcript": audio["transcript"], **result}
