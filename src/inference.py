"""Load the saved WhisperWatch pipeline and score Whisper measurements."""

from math import isfinite
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_NAMES, finite_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "whisperwatch.joblib"
MODEL_FORMAT = 1


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load a local joblib artifact produced by src.train."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model missing at {path}. Run python -m src.train dev first.")
    artifact = joblib.load(path)
    if (
        artifact.get("format_version") != MODEL_FORMAT
        or tuple(artifact.get("feature_names", ())) != FEATURE_NAMES
        or tuple(artifact["pipeline"].feature_names_in_) != FEATURE_NAMES
    ):
        raise ValueError("The saved feature schema does not match this code. Retrain with src.train.")
    return artifact


def score_features(artifact: dict, rows) -> list[float]:
    """Score a batch using the saved schema and the high-error class (1)."""
    names = artifact["feature_names"]
    frame = pd.DataFrame(
        ([finite_or_nan(row.get(name)) for name in names] for row in rows),
        columns=names,
    )
    pipeline = artifact["pipeline"]
    class_index = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(frame)[:, class_index].tolist()


def predict_from_features(features: dict) -> dict:
    """Return a review-risk score, recommendation, and JSON-safe measurements."""
    artifact = load_model()
    score = score_features(artifact, [features])[0]
    threshold = artifact["review_threshold"]
    details = {}
    for name in artifact["feature_names"]:
        value = finite_or_nan(features.get(name))
        details[name] = value if isfinite(value) else None
    details["review_threshold"] = threshold
    return {
        "review_score": score,
        "review_label": "Review recommended" if score >= threshold else "Low predicted risk",
        "details": details,
    }
