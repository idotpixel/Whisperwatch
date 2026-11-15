"""Call the team's backend lazily and translate failures into UI messages."""

import logging
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path

from ui.replay import number_field, text_field

LOGGER = logging.getLogger(__name__)
OFFLINE = "Live analysis needs the backend and trained model. Dataset Replay is available."


class UploadError(Exception):
    """An error whose message can be shown to the user."""


def backend_present():
    """Check backend and model availability without loading Whisper or classifier weights."""
    if find_spec("src.asr") is None:
        return False
    try:
        from src.inference import PRODUCTION_MODEL_PATH
    except ImportError:
        return False
    return PRODUCTION_MODEL_PATH.is_file()


def analyze_upload(audio_path):
    """Pass a filepath through to ASR and validate the fields the UI uses."""
    from src.asr import analyze_audio

    result = analyze_audio(str(audio_path))
    if not isinstance(result, Mapping) or not isinstance(result.get("details"), Mapping):
        raise TypeError("Expected an analysis dictionary with details.")
    return {
        "transcript": text_field(result.get("transcript"), "transcript", allow_empty=True),
        "review_score": number_field(result.get("review_score"), "review_score"),
        "review_label": text_field(result.get("review_label"), "review_label"),
        "details": dict(result["details"]),
    }


def analyze_batch(audio_paths, progress):
    """Score every clip before asking the backend to rank the complete batch."""
    from src.inference import rank_recordings

    analyzed = []
    filenames = {}
    total = len(audio_paths)
    for index, path in enumerate(audio_paths, 1):
        clip_id = f"clip-{index:04d}"
        filename = Path(path).name
        progress((index - 1, total), desc=f"Analyzing clip {index} of {total}")
        try:
            analyzed.append(dict(analyze_upload(path), clip_id=clip_id))
        except Exception as error:
            raise UploadError(f"{filename} ({clip_id}): {error_message(error)}") from error
        filenames[clip_id] = filename
    progress(0.95, desc="Ranking the batch")
    return rank_recordings(analyzed, method="hybrid"), filenames


def error_message(error):
    """Report common integration and media errors without exposing paths or tracebacks."""
    if isinstance(error, UploadError):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return "A required file is missing. Re-upload the recording and check the model setup."
    if isinstance(error, ImportError):
        return "The backend could not start. Check its Python and runtime dependencies."
    if isinstance(error, MemoryError) or (
        isinstance(error, RuntimeError)
        and any(marker in str(error).lower() for marker in ("out of memory", "failed to allocate memory"))
    ):
        return "Whisper ran out of memory. Close memory-heavy apps and retry, or use the Colab GPU setup."
    if isinstance(error, ValueError):
        if "up to 30 seconds" in str(error):
            return "The recording exceeds 30 seconds. Trim it and try again."
        if "empty" in str(error) or "positive duration" in str(error):
            return "The recording contains no playable audio."
    return "Analysis failed. Try a playable English clip up to 30 seconds, or check the backend and model setup."
