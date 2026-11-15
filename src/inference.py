"""Load the saved WhisperWatch pipeline and score Whisper measurements."""

from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version

from .features import FEATURE_NAMES, finite_or_nan
from .ranking import blend, ranked_ids, review_queue, risk_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "whisperwatch.joblib"
MODEL_FORMAT = 1
PRODUCTION_MODEL_PATH = MODEL_PATH.with_name("whisperwatch_boosted.joblib")
PRODUCTION_MODEL_MEMBER = "models/whisperwatch_boosted.joblib"
PRODUCTION_MODEL_SHA256 = "8d197a7d191a63db1fbbdf31bceaf0badea2b07a88b8df756ccd7283bb6dc496"
PRODUCTION_SKLEARN_VERSION = "1.8.0"
RANKING_METHODS = ("hybrid", "native_confidence", "classifier")


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


def _check_production_bytes(payload: bytes) -> None:
    if sha256(payload).hexdigest() != PRODUCTION_MODEL_SHA256:
        raise ValueError("Expected the original frozen boosted model from whisperwatch-boosted-results.zip.")


def install_production_model(bundle, path: Path = PRODUCTION_MODEL_PATH) -> Path:
    """Extract the approved member from the original ZIP after validating its bytes."""
    with ZipFile(bundle) as archive:
        with archive.open(PRODUCTION_MODEL_MEMBER) as model:
            payload = model.read(1_000_001)
    _check_production_bytes(payload)
    path = Path(path)
    if path.exists():
        _check_production_bytes(path.read_bytes())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(payload)
    return path


@lru_cache(maxsize=1)
def load_production_model(path: Path = PRODUCTION_MODEL_PATH) -> dict:
    """Verify the release checksum before deserializing and cache the fitted model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model missing at {path}. Install it from whisperwatch-boosted-results.zip.")
    payload = path.read_bytes()
    _check_production_bytes(payload)
    if sklearn_version != PRODUCTION_SKLEARN_VERSION:
        raise RuntimeError(f"Install scikit-learn=={PRODUCTION_SKLEARN_VERSION} for the frozen model.")
    artifact = joblib.load(BytesIO(payload))
    fitted = artifact["pipeline"]
    if (artifact.get("format_version") != MODEL_FORMAT or tuple(artifact["feature_names"]) != FEATURE_NAMES
            or tuple(fitted.feature_names_in_) != FEATURE_NAMES or list(fitted.classes_) != [0, 1]):
        raise ValueError("The frozen model schema or high-error class is incompatible.")
    return artifact


def predict_from_features(features: dict, *, model_path: Path = PRODUCTION_MODEL_PATH) -> dict:
    """Return a single-clip score and contextual guidance for the original model."""
    artifact = load_production_model(model_path)
    score = score_features(artifact, [features])[0]
    details = {}
    for name in FEATURE_NAMES:
        number = finite_or_nan(features.get(name))
        details[name] = number if isfinite(number) else None
    details["score_type"] = "single_clip_classifier_risk"
    return {"review_score": score, "review_label": "Compare in batch for review priority", "details": details}


def rank_recordings(recordings, *, method: str = "hybrid") -> list[dict]:
    """Rank analyzed clips; each row needs clip_id, review_score and details.avg_logprob."""
    if method not in RANKING_METHODS:
        raise ValueError(f"Choose one ranking method: {', '.join(RANKING_METHODS)}.")
    by_id, inputs = {}, []
    for recording in recordings:
        if not isinstance(recording, Mapping):
            raise ValueError("Each recording must be a dictionary of analyzed clip data.")
        clip_id = recording.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip() or clip_id in by_id:
            raise ValueError("Every recording needs a unique nonempty string clip_id.")
        score = finite_or_nan(recording.get("review_score"))
        if not 0 <= score <= 1:
            raise ValueError(f"Missing or invalid review_score for {clip_id}.")
        details = recording.get("details")
        if not isinstance(details, Mapping) or "avg_logprob" not in details:
            raise ValueError(f"Missing details.avg_logprob field for {clip_id}. Use None for missing confidence.")
        by_id[clip_id] = recording
        inputs.append((clip_id, score, finite_or_nan(details["avg_logprob"])))
    if not inputs:
        return []
    frame = pd.DataFrame(inputs, columns=["clip_id", "risk_score", "avg_logprob"]).set_index("clip_id")
    signals = risk_signals(frame)
    signals["hybrid"] = blend(signals, .50)
    key = {"hybrid": "hybrid", "native_confidence": "whisper_confidence", "classifier": "boosted_trees"}[method]
    selected = set(review_queue(signals[key]))
    results = []
    for rank, clip_id in enumerate(ranked_ids(signals[key]), start=1):
        row = signals.loc[clip_id]
        native = float(row.whisper_confidence)
        results.append({"clip_id": clip_id, "transcript": by_id[clip_id].get("transcript", ""),
                        "classifier_risk": float(row.boosted_trees),
                        "native_confidence_risk": native if isfinite(native) else None,
                        "hybrid_risk": float(row.hybrid), "ranking_method": method, "rank": rank,
                        "selected_for_review": clip_id in selected})
    return results


def predict_legacy_from_features(features: dict, *, model_path: Path = MODEL_PATH) -> dict:
    """Retain the historical logistic notebook's archived-cutoff demonstration."""
    """Return a review-risk score, recommendation, and JSON-safe measurements."""
    artifact = load_model(model_path)
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
