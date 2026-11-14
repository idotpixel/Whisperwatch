"""Train on English Whispered metadata and export development or test scores.

Run `python -m src.train dev` to fit the initial model and choose its review
threshold, then `python -m src.train test` to score the held-out split once
the model and evaluation procedure are frozen.
"""

import argparse
from math import isfinite
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import joblib
import pandas as pd
import pyarrow.parquet as pq
import sklearn
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
TEACHER = "openai/whisper-large-v3"
ARCHIVE_PIPELINE = "0.2.0"
MAX_DURATION_S = 30
WER_THRESHOLD = 0.20
SEED = 42
TRAIN_ROWS_PER_SHARD = 5000
SHARDS = {"train": (0, 13, 26), "dev": (0,), "test": (0,)}
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


def prepare_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read English clips from the selected shards and construct features and labels."""
    frames = []
    for shard in SHARDS[split]:
        path = get_shard(split, shard, data_dir)
        frame = pq.read_table(path, columns=list(DATA_COLUMNS)).to_pandas()
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
        if split == "train":
            eligible = eligible.sample(n=min(TRAIN_ROWS_PER_SHARD, len(eligible)), random_state=SEED)
        frames.append(eligible)

    data = pd.concat(frames, ignore_index=True).drop_duplicates("clip_id").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No usable English examples in {split}.")
    # Reference text and WER never reach extract_features or the fitted pipeline.
    features = pd.DataFrame(
        (extract_features(segments, transcript, duration) for segments, transcript, duration in
         data[["segments", "whisper_text", "duration_s"]].itertuples(index=False, name=None)),
        columns=FEATURE_NAMES,
    )
    data = data.rename(columns={"language_bcp47": "language", "wer_whisper_vs_human": "wer"})
    data["true_high_error"] = (data["wer"] > WER_THRESHOLD).astype(int)
    return pd.concat([data[["clip_id", "language", "true_high_error", "wer", "whisper_text", "human_text"]], features], axis=1)


def write_predictions(split: str, data: pd.DataFrame, scores: list[float]) -> Path:
    """Export risk scores and the ground truth needed by the evaluation team."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{split}_predictions.csv"
    data.assign(risk_score=scores).to_csv(path, columns=PREDICTION_COLUMNS, index=False)
    print(f"Saved {len(data):,} predictions to {path}")
    return path


def train_development() -> None:
    """Fit a fixed logistic regression on train and choose an F1 cutoff on dev."""
    train = prepare_split("train")
    dev = prepare_split("dev")
    for split, data in (("train", train), ("dev", dev)):
        labels = data["true_high_error"]
        if labels.nunique() != 2:
            raise ValueError(f"The {split} split needs both error classes.")
        print(f"{split}: {len(data):,} rows, {labels.sum():,} high-error clips")

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipeline.fit(train[list(FEATURE_NAMES)], train["true_high_error"])
    artifact = {
        "format_version": MODEL_FORMAT,
        "feature_names": FEATURE_NAMES,
        "pipeline": pipeline,
        "dataset_revision": REVISION,
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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    reloaded = load_model()
    reloaded_scores = score_features(reloaded, dev[list(FEATURE_NAMES)].to_dict("records"))
    if scores != reloaded_scores:
        raise RuntimeError("Reloading the model changed development scores.")
    print(f"Dev average precision: {average_precision_score(labels, scores):.4f}")
    print(f"Dev F1: {f1:.4f} at review threshold {threshold:.6f}")
    print(f"Saved and reloaded {MODEL_PATH}")
    write_predictions("dev", dev, reloaded_scores)


def predict_test() -> None:
    """Use the existing artifact and development threshold on official test clips."""
    artifact = load_model()
    if artifact["dataset_revision"] != REVISION or artifact["wer_threshold"] != WER_THRESHOLD:
        raise ValueError("Artifact dataset or label differs from the frozen evaluation procedure.")
    test = prepare_split("test")
    scores = score_features(artifact, test[list(FEATURE_NAMES)].to_dict("records"))
    write_predictions("test", test, scores)
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
