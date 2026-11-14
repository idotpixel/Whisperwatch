"""Train on English Whispered metadata and export development or test scores.

In Colab, train_development(full_training=True) uses all eligible English
training shards. The earlier 15,000-clip configuration remains the default
for reproducing the preliminary experiment. Score the held-out split once
the model and evaluation procedure are frozen.
"""

import argparse
from math import ceil, isfinite
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features, finite_or_nan
from .inference import MODEL_FORMAT, MODEL_PATH, PROJECT_ROOT, load_model, score_features


DATASET = "burakaydinofficial/Whispered"
REVISION = "25299fd560dde4271f5177118354af5a961b92fd"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "whispered"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BOOSTED_MODEL_PATH = MODEL_PATH.with_name("whisperwatch_boosted.joblib")
BOOSTED_OUTPUT_DIR = OUTPUT_DIR / "boosted"
TEACHER = "openai/whisper-large-v3"
ARCHIVE_PIPELINE = "0.2.0"
MAX_DURATION_S = 30
WER_THRESHOLD = 0.20
SEED = 42
TRAIN_ROWS_PER_SHARD = 5000
SHARDS = {"train": (0, 13, 26), "dev": (0,), "test": (0,)}
FULL_TRAIN_SHARDS = tuple(range(28))
DATA_COLUMNS = (
    "clip_id", "language_bcp47", "split", "teacher_model", "runner", "compute_type",
    "pipeline_version", "duration_s", "segments", "wer_whisper_vs_human",
    "whisper_text", "human_text",
)
PREDICTION_COLUMNS = (
    "clip_id", "language", "true_high_error", "risk_score", "avg_logprob", "wer",
    "whisper_text", "human_text",
)


def get_shard(split: str, shard: int, data_dir: Path) -> Path:
    """Cache one Parquet file from the pinned English release."""
    name = f"{split}-{shard:05d}.parquet"
    path = data_dir / name
    if path.is_file():
        return path

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/common_voice_17/en/{split}/{name}"
    data_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=data_dir, delete=False) as partial:
        temporary_path = Path(partial.name)
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as target:
            copyfileobj(response, target)
        pq.read_schema(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def prepare_split(
    split: str, data_dir: Path = DATA_DIR, *, full_training: bool = False,
    feature_extractor=extract_features,
) -> pd.DataFrame:
    """Read English clips from the selected shards and construct features and labels."""
    training = split == "train"
    shards = FULL_TRAIN_SHARDS if training and full_training else SHARDS[split]
    columns = [name for name in DATA_COLUMNS if not (training and name == "human_text")]
    frames = []
    for shard in shards:
        path = get_shard(split, shard, data_dir)
        frame = pq.read_table(path, columns=columns).to_pandas()
        if not frame["language_bcp47"].eq("en").all() or not frame["split"].eq(split).all():
            raise ValueError(f"Unexpected language or split in {path}")
        eligible = frame.loc[
            frame["teacher_model"].eq(TEACHER)
            & frame["runner"].eq("faster-whisper")
            & frame["compute_type"].eq("float16")
            & frame["pipeline_version"].eq(ARCHIVE_PIPELINE)
            & frame["duration_s"].between(0, MAX_DURATION_S, inclusive="right")
            & frame["wer_whisper_vs_human"].map(lambda value: isfinite(finite_or_nan(value)) and value >= 0)
        ]
        print(f"{path.name}: {len(eligible):,} of {len(frame):,} rows eligible", flush=True)
        if training and not full_training:
            eligible = eligible.sample(n=min(TRAIN_ROWS_PER_SHARD, len(eligible)), random_state=SEED)
        if eligible.empty:
            continue
        eligible = eligible.reset_index(drop=True)
        # Keep just the numeric features and labels from each training shard in memory.
        features = pd.DataFrame.from_records(
            (feature_extractor(segments, transcript, duration) for segments, transcript, duration in
             eligible[["segments", "whisper_text", "duration_s"]].itertuples(index=False, name=None)),
        )
        features["clip_id"] = eligible["clip_id"]
        features["true_high_error"] = (eligible["wer_whisper_vs_human"] > WER_THRESHOLD).astype(int)
        if not training:
            features["language"] = eligible["language_bcp47"]
            features["wer"] = eligible["wer_whisper_vs_human"]
            features["whisper_text"] = eligible["whisper_text"]
            features["human_text"] = eligible["human_text"]
        frames.append(features)

    if not frames:
        raise ValueError(f"No usable English examples in {split}.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("clip_id").reset_index(drop=True)


def write_predictions(
    split: str, data: pd.DataFrame, scores: list[float], output_dir: Path = OUTPUT_DIR
) -> Path:
    """Export risk scores and the ground truth needed by the evaluation team."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_predictions.csv"
    data.assign(risk_score=scores).to_csv(path, columns=PREDICTION_COLUMNS, index=False)
    print(f"Saved {len(data):,} predictions to {path}")
    return path


def train_development(*, full_training: bool = False) -> None:
    """Fit a fixed logistic regression on train and choose an F1 cutoff on dev."""
    train = prepare_split("train", full_training=full_training)
    dev = prepare_split("dev")
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    _fit_and_export(pipeline, train, dev, full_training, MODEL_PATH, OUTPUT_DIR)


def train_boosted_development(baseline_zip: Path) -> None:
    """Fit boosted trees on the full English training set and compare development rankings."""
    dev = prepare_split("dev")
    # Compare the exact same clips and labels against the saved full logistic run.
    columns = ["clip_id", "true_high_error", "risk_score"]
    with ZipFile(baseline_zip) as archive, archive.open("outputs/dev_predictions.csv") as csv:
        baseline = pd.read_csv(csv, usecols=columns).set_index("clip_id")
    expected = dev.set_index("clip_id")["true_high_error"]
    if not baseline.index.is_unique or set(baseline.index) != set(expected.index):
        raise ValueError("The logistic ZIP must contain predictions for the same development clips.")
    baseline = baseline.reindex(expected.index)
    if not baseline["true_high_error"].eq(expected).all() or not baseline["risk_score"].between(0, 1).all():
        raise ValueError("The logistic ZIP contains different labels or invalid risk scores.")

    train = prepare_split("train", full_training=True)
    classifier = make_boosted_classifier()
    scores = _fit_and_export(
        classifier, train, dev, True, BOOSTED_MODEL_PATH, BOOSTED_OUTPUT_DIR
    )
    comparison = compare_review_rankings(dev, baseline["risk_score"].tolist(), scores)
    comparison.to_csv(BOOSTED_OUTPUT_DIR / "dev_comparison.csv", index=False)
    print("\nDevelopment comparison at the same 20% review budget")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def make_boosted_classifier() -> HistGradientBoostingClassifier:
    """Use the same tree settings for the original and feature experiments."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        early_stopping=True,
        random_state=SEED,
    )


def compare_review_rankings(dev: pd.DataFrame, logistic_scores: list, boosted_scores: list) -> pd.DataFrame:
    """Count high-error clips found by each ranking with an identical review budget."""
    # Rank missing Whisper confidence first, matching the earlier baseline comparison.
    rankings = dev.assign(
        logistic_regression=logistic_scores,
        boosted_trees=boosted_scores,
        whisper_confidence=(-dev["avg_logprob"]).fillna(float("inf")),
    )
    budget = ceil(len(rankings) * 0.20)
    labels = rankings["true_high_error"]
    high_error = int(labels.sum())
    rows = []
    for method in ("boosted_trees", "logistic_regression", "whisper_confidence"):
        ranked = rankings.sort_values([method, "clip_id"], ascending=[False, True])
        caught = int(ranked["true_high_error"].iloc[:budget].sum())
        # sklearn's average precision requires finite inputs and only uses score order.
        ap_scores = rankings[method].rank(method="dense")
        rows.append({
            "method": method,
            "clips_reviewed": budget,
            "high_error_caught": caught,
            "high_error_missed": high_error - caught,
            "recall": caught / high_error,
            "precision": caught / budget,
            "average_precision": average_precision_score(labels, ap_scores),
        })
    return pd.DataFrame(rows)


def _fit_and_export(pipeline, train, dev, full_training, model_path, output_dir) -> list[float]:
    for split, data in (("train", train), ("dev", dev)):
        labels = data["true_high_error"]
        if labels.nunique() != 2:
            raise ValueError(f"The {split} split needs both error classes.")
        print(f"{split}: {len(data):,} rows, {labels.sum():,} high-error clips")

    pipeline.fit(train[list(FEATURE_NAMES)], train["true_high_error"])
    artifact = {
        "format_version": MODEL_FORMAT,
        "feature_names": FEATURE_NAMES,
        "pipeline": pipeline,
        "dataset_revision": REVISION,
        "training_rows": len(train),
        "training_shards": FULL_TRAIN_SHARDS if full_training else SHARDS["train"],
        "training_rows_per_shard": None if full_training else TRAIN_ROWS_PER_SHARD,
        "wer_threshold": WER_THRESHOLD,
        "sklearn_version": sklearn.__version__,
    }
    scores = score_features(artifact, dev[list(FEATURE_NAMES)].to_dict("records"))
    labels = dev["true_high_error"]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Use the higher threshold if several cutoffs give the same best F1.
    f1, threshold = max(
        (2 * p * r / (p + r), cutoff)
        for p, r, cutoff in zip(precision, recall, thresholds)
        if p + r > 0
    )
    artifact["review_threshold"] = float(threshold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    reloaded = load_model(model_path)
    reloaded_scores = score_features(reloaded, dev[list(FEATURE_NAMES)].to_dict("records"))
    if scores != reloaded_scores:
        raise RuntimeError("Reloading the model changed development scores.")
    print(f"Dev average precision: {average_precision_score(labels, scores):.4f}")
    print(f"Dev F1: {f1:.4f} at review threshold {threshold:.6f}")
    print(f"Saved and reloaded {model_path}")
    write_predictions("dev", dev, reloaded_scores, output_dir)
    return reloaded_scores


def predict_test(*, model_path: Path = MODEL_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    """Use the existing artifact and development threshold on official test clips."""
    artifact = load_model(model_path)
    if artifact["dataset_revision"] != REVISION or artifact["wer_threshold"] != WER_THRESHOLD:
        raise ValueError("Artifact dataset or label differs from the frozen evaluation procedure.")
    test = prepare_split("test")
    scores = score_features(artifact, test[list(FEATURE_NAMES)].to_dict("records"))
    write_predictions("test", test, scores, output_dir)
    predicted = [score >= artifact["review_threshold"] for score in scores]
    print(f"Test average precision: {average_precision_score(test['true_high_error'], scores):.4f}")
    print(f"Test F1 at saved threshold: {f1_score(test['true_high_error'], predicted):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "test"))
    args = parser.parse_args()
    if args.split == "dev":
        train_development()
    else:
        predict_test()
