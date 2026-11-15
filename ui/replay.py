"""Load Member C's recorded predictions for Dataset Replay."""

import json
from dataclasses import dataclass
from pathlib import Path


def read_json(path):
    """Read a UTF-8 JSON evaluation asset."""
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def read_object(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def text_field(value, name, *, allow_empty=False):
    """Validate a text field and keep the supplied transcript intact."""
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a string{' with content' if not allow_empty else ''}.")
    return value


def number_field(value, name, *, maximum=1):
    """Validate a nonnegative numeric measurement and reject non-finite values."""
    if (
        type(value) not in (int, float)
        or not 0 <= value < float("inf")
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a finite number in the documented range.")
    return float(value)


def review_label(value):
    """Accept the two Boolean labels or their documented text equivalents."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ("review recommended", "low predicted risk"):
            return label == "review recommended"
    raise ValueError("review_label must be a Boolean or a supported recommendation.")


@dataclass(frozen=True)
class Example:
    clip_id: str
    dataset: str
    risk_score: float
    recommend_review: bool
    true_high_error: bool
    wer: float
    whisper_text: str
    human_text: str
    classifier_risk: float | None = None
    avg_logprob: float | None = None
    outcome_category: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Replay:
    examples: dict[str, Example]
    source: str
    notice: str = ""


def parse_replay(document):
    """Build the replay collection without using ground truth as a prediction."""
    if isinstance(document, list):
        rows, source, threshold = document, "Member C’s curated evaluation cases", None
    elif isinstance(document, dict):
        rows = document.get("examples")
        source = text_field(document.get("source"), "source")
        threshold = document.get("review_threshold")
    else:
        raise ValueError("Expected a list of clips or an export object.")
    if threshold is not None:
        threshold = number_field(threshold, "review_threshold")
    if not isinstance(rows, list) or not rows:
        raise ValueError("examples must be a nonempty list.")

    examples = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each example must be an object.")
        clip_id = text_field(row.get("clip_id"), "clip_id")
        if clip_id in examples:
            raise ValueError(f"Duplicate clip_id: {clip_id}.")
        score = number_field(row.get("risk_score"), "risk_score")
        if "recommend_review" in row:
            predicted = review_label(row["recommend_review"])
        elif "review_label" in row:
            predicted = review_label(row["review_label"])
        elif threshold is not None:
            predicted = score >= threshold
        else:
            raise ValueError("Provide review_label for each clip or a review_threshold.")
        truth = row.get("true_high_error")
        if type(truth) not in (bool, int) or truth not in (0, 1):
            raise ValueError("true_high_error must be a Boolean or a 0/1 label.")
        classifier = row.get("classifier_risk")
        confidence = row.get("avg_logprob")
        if confidence is not None and (
            type(confidence) not in (float, int) or not -float("inf") < confidence <= 0
        ):
            raise ValueError("avg_logprob must be finite and nonpositive.")
        examples[clip_id] = Example(
            clip_id,
            text_field(row.get("dataset", row.get("language")), "dataset"),
            score,
            predicted,
            bool(truth),
            number_field(row.get("wer"), "wer", maximum=None),
            text_field(row.get("whisper_text"), "whisper_text", allow_empty=True),
            text_field(row.get("human_text"), "human_text", allow_empty=True),
            None if classifier is None else number_field(classifier, "classifier_risk"),
            confidence,
            text_field(row.get("outcome_category", ""), "outcome_category", allow_empty=True),
            text_field(row.get("notes", ""), "notes", allow_empty=True),
        )
    return Replay(examples, source)


def load_replay(path: Path):
    """Load the held-out export, or describe why replay is unavailable."""
    try:
        return parse_replay(read_json(path))
    except FileNotFoundError:
        return Replay({}, "", "No held-out replay examples are available yet.")
    except (OSError, ValueError):
        return Replay({}, "", "The held-out replay export could not be loaded.")
