"""Compare fixed blends of Whisper and boosted-tree risk on development clips."""

from math import ceil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_NAMES, extract_features
from .train import DATA_DIR, get_shard, make_boosted_classifier, prepare_split


BOOSTED_WEIGHTS = (0.25, 0.50, 0.75)
SPLIT_SEED = 20261114
REVIEW_FRACTION = 0.20
LABEL = "true_high_error"
DEV_MEMBER = "outputs/boosted/dev_predictions.csv"


def read_development(bundle) -> pd.DataFrame:
    """Read only the clip IDs, two risk signals, and evaluation labels."""
    columns = ["clip_id", "language", LABEL, "avg_logprob", "risk_score"]
    with ZipFile(bundle) as archive, archive.open(DEV_MEMBER) as csv:
        dev = pd.read_csv(csv, usecols=columns).set_index("clip_id").sort_index()
    if (
        dev.empty
        or not dev.index.is_unique
        or not dev.index.str.startswith("common_voice_17/en/dev/").all()
        or not dev["language"].eq("en").all()
        or set(dev[LABEL]) != {0, 1}
        or not dev["risk_score"].between(0, 1).all()
        or not np.isfinite(dev["avg_logprob"].dropna()).all()
    ):
        raise ValueError("Expected unique English development clips with valid risk scores and binary labels.")
    return dev


def risk_signals(dev: pd.DataFrame) -> pd.DataFrame:
    """Make higher values riskier for both methods."""
    return pd.DataFrame({
        "boosted_trees": dev["risk_score"],
        # Missing confidence goes to the front of the original baseline queue.
        "whisper_confidence": -dev["avg_logprob"].fillna(-np.inf),
    })


def review_queue(risk: pd.Series) -> pd.Index:
    """Take the highest-risk 20%, breaking ties by ascending clip ID."""
    ranked = risk.sort_index().sort_values(ascending=False, kind="stable")
    return ranked.index[:ceil(len(ranked) * REVIEW_FRACTION)]


def detected_overlap(dev: pd.DataFrame) -> dict:
    """Count the high-error clips caught by each queue on all development data."""
    signals = risk_signals(dev)
    positive = dev.index[dev[LABEL].eq(1)]
    boosted = set(review_queue(signals["boosted_trees"]).intersection(positive))
    native = set(review_queue(signals["whisper_confidence"]).intersection(positive))
    return {
        "clips": len(dev),
        "clips_reviewed_per_method": ceil(len(dev) * REVIEW_FRACTION),
        "high_error_clips": len(positive),
        "caught_by_both": len(boosted & native),
        "caught_only_by_trees": len(boosted - native),
        "caught_only_by_whisper": len(native - boosted),
        "caught_by_neither": len(positive) - len(boosted | native),
    }


def review_metrics(labels: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each signal on identical clip IDs at the same budget."""
    if not labels.index.equals(signals.index):
        raise ValueError("The rankings and labels must contain the same clips in the same order.")
    rows = []
    high_error = int(labels.sum())
    for method, scores in signals.items():
        queue = review_queue(scores)
        caught = int(labels.loc[queue].sum())
        rows.append({
            "method": method,
            "reviewed": len(queue),
            "caught": caught,
            "precision_at_budget": caught / len(queue),
            "recall_at_budget": caught / high_error,
        })
    results = pd.DataFrame(rows).set_index("method")
    results["extra_vs_trees"] = results["caught"] - results.at["boosted_trees", "caught"]
    results["extra_vs_whisper"] = results["caught"] - results.at["whisper_confidence", "caught"]
    return results


def blend(signals: pd.DataFrame, boosted_weight: float) -> pd.Series:
    """Average percentile ranks with ties receiving their average rank."""
    ranks = signals.rank(method="average", pct=True)
    return boosted_weight * ranks["boosted_trees"] + (1 - boosted_weight) * ranks["whisper_confidence"]


def split_by_speaker(dev: pd.DataFrame, speakers: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign whole speakers to tuning or verification using the fixed seed."""
    if not speakers.index.is_unique:
        raise ValueError("The archive has duplicate clip IDs in its speaker metadata.")
    groups = speakers.reindex(dev.index)
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Every development clip needs an archived speaker ID.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SPLIT_SEED)
    tuning_indices, verification_indices = next(splitter.split(dev, groups=groups))
    tuning, verification = dev.iloc[tuning_indices], dev.iloc[verification_indices]
    if tuning[LABEL].nunique() != 2 or verification[LABEL].nunique() != 2:
        raise ValueError("Both development partitions need both error classes.")
    return tuning, verification


def compare_ensemble(dev: pd.DataFrame):
    """Select on tuning speakers and score the chosen blend on verification speakers."""
    path = get_shard("dev", 0, DATA_DIR)
    metadata = pq.read_table(path, columns=["clip_id", "speaker_id"]).to_pandas()
    tuning, verification = split_by_speaker(dev, metadata.set_index("clip_id")["speaker_id"])
    tuning_signals = risk_signals(tuning)
    # Prefer the middle weight on ties, then the lower tree weight.
    candidates = {weight: f"blend_trees_{weight:.2f}" for weight in (0.50, 0.25, 0.75)}
    for weight in BOOSTED_WEIGHTS:
        tuning_signals[candidates[weight]] = blend(tuning_signals, weight)
    tuning_results = review_metrics(tuning[LABEL], tuning_signals)
    selected_weight = max(candidates, key=lambda weight: tuning_results.at[candidates[weight], "caught"])

    verification_signals = risk_signals(verification)
    verification_signals["selected_blend"] = blend(verification_signals, selected_weight)
    verification_results = review_metrics(verification[LABEL], verification_signals)
    return selected_weight, tuning, verification, tuning_results, verification_results


def extract_extended_features(segments, transcript: str, duration_s: float) -> dict:
    """Add transcript repetition and word-length measurements available for new clips."""
    features = extract_features(segments, transcript, duration_s)
    words = transcript.casefold().split() if isinstance(transcript, str) else []
    pairs = list(zip(words, words[1:]))
    features["repeated_word_fraction"] = 1 - len(set(words)) / len(words) if words else 0.0
    features["repeated_bigram_fraction"] = 1 - len(set(pairs)) / len(pairs) if pairs else 0.0
    features["mean_word_length"] = sum(map(len, words)) / len(words) if words else 0.0
    return features


def compare_added_features(tuning: pd.DataFrame) -> pd.DataFrame:
    """Fit one Colab-only candidate using three extra inference-time features."""
    columns = list(extract_extended_features([], "", 0))
    assert tuple(columns[:len(FEATURE_NAMES)]) == FEATURE_NAMES
    archive_dev = prepare_split("dev", feature_extractor=extract_extended_features).set_index("clip_id")
    scored_dev = archive_dev.loc[tuning.index]
    if not scored_dev[LABEL].equals(tuning[LABEL]):
        raise ValueError("The archive development labels differ from the boosted export.")
    train = prepare_split("train", full_training=True, feature_extractor=extract_extended_features)
    classifier = make_boosted_classifier()
    classifier.fit(train[columns], train[LABEL])
    scores = classifier.predict_proba(scored_dev[columns])[:, list(classifier.classes_).index(1)]
    signals = risk_signals(tuning).assign(boosted_with_added_features=scores)
    return review_metrics(tuning[LABEL], signals)
