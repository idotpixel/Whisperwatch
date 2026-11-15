"""Export and reconcile existing result records for the Member C handoff."""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .ranking import blend, ranked_ids, risk_signals


BOOSTED = "whisperwatch-boosted-results.zip"
SPLIT = "whisperwatch-ensemble-speaker-split-results.zip"
LIBRISPEECH = "whisperwatch-live-validation-results.zip"
AMI_RETRAIN = "whisperwatch-live-retraining-results.zip"
AMI_AUDIO = "whisperwatch-audio-aware-results.zip"


def _read(bundle, name):
    with bundle.open(name) as file:
        return pd.read_csv(file, float_precision="round_trip", keep_default_na=False, na_values=[])


def _common_voice(source, development_metadata):
    with ZipFile(source / BOOSTED) as model, ZipFile(source / SPLIT) as split:
        protocol = json.loads(split.read("protocol_and_recommendation.json"))
        if protocol["selected_tree_weight"] != .50 or protocol["review_fraction"] != .20:
            raise ValueError("Unexpected archived ranking protocol.")
        dev = _read(model, "outputs/boosted/dev_predictions.csv").set_index("clip_id", verify_integrity=True)
        split_map = _read(split, "development_split.csv").set_index("clip_id", verify_integrity=True)
        if set(dev.index) != set(split_map.index) or not dev.index.str.startswith("common_voice_17/en/dev/").all():
            raise ValueError("The archived predictions must match the English development manifest.")
        saved = _read(split, "verification_results.csv").set_index("method")
    ids = split_map.index[split_map.partition.eq("verification")]
    rows = dev.loc[ids].sort_index().rename(columns={"whisper_text": "transcript", "human_text": "reference"})
    metadata = pq.read_table(development_metadata, columns=["clip_id", "speaker_id", "split"]).to_pandas()
    if not metadata.split.eq("dev").all():
        raise ValueError("Only Whispered development metadata is permitted.")
    speakers = metadata.set_index("clip_id", verify_integrity=True).speaker_id
    if not set(speakers.loc[ids]).isdisjoint(speakers.loc[split_map.index[split_map.partition.eq("tuning")]]):
        raise ValueError("Archived tuning and verification speakers overlap.")
    rows["speaker_id"] = speakers.loc[rows.index]
    return rows, saved, {"classifier": "boosted_trees", "native": "whisper_confidence", "original_hybrid": "selected_blend"}


def _live_records(source, archive_name, member, summary, prefix, tree, hybrid, label, variants=(), has_random=False):
    with ZipFile(source / archive_name) as bundle:
        records = _read(bundle, member).set_index("clip_id", verify_integrity=True).sort_index()
        saved = _read(bundle, summary)
        if "classifier" in saved:
            saved = saved.loc[saved.classifier.eq("archived_training")]
        saved = saved.set_index("method", verify_integrity=True)
    frame = records.rename(columns={label: "true_high_error"}).copy()
    native = pd.to_numeric(records[prefix + "score_native_confidence"])
    rows = {"native": "native_confidence", "original_hybrid": hybrid}
    if tree is not None:
        rows["classifier"] = tree
    if has_random:
        rows["random"] = "random"
    rows.update({name: name for name in variants})
    provided = {name: (pd.to_numeric(records[prefix + "score_" + actual]), records[prefix + "review_" + actual])
                for name, actual in rows.items()}
    # The audio-aware run saved the original hybrid but omitted the original tree score.
    rows.setdefault("classifier", None)
    invariant = -pd.to_numeric(frame.avg_logprob.replace("", np.nan)).fillna(-np.inf)
    np.testing.assert_allclose(native, invariant, rtol=1e-12, atol=1e-12)
    return frame, saved, rows, provided


def _file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export(source, development_metadata, output):
    """Rebuild the approved rankings and reconcile them with the existing exports."""
    from .features import FEATURE_NAMES
    from .inference import install_production_model, load_production_model, score_features
    from tempfile import TemporaryDirectory

    common, cm_saved, cm_methods = _common_voice(source, development_metadata)
    lib = _live_records(source, LIBRISPEECH, "predictions.csv", "budget_comparison.csv", "", "boosted_trees",
                        "hybrid_50_50", "true_high_error", has_random=True)
    ami = _live_records(source, AMI_RETRAIN, "held_out_predictions.csv", "comparison.csv", "archived_training_",
                        "boosted_trees", "hybrid_50_50", "high_error", has_random=True)
    final = _live_records(source, AMI_AUDIO, "evaluation_predictions.csv", "comparison.csv", "", None,
                          "original_hybrid", "high_error", variants=("control", "audio_aware", "control_hybrid", "audio_hybrid"))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        model_path = install_production_model(source / BOOSTED, Path(temporary) / "boosted.joblib")
        artifact = load_production_model(model_path)
        for frame, _, _, _ in (lib, ami, final):
            features = frame[list(FEATURE_NAMES)].replace("", np.nan)
            frame["risk_score"] = score_features(artifact, features.to_dict("records"))
    all_metrics = []
    specs = (
        ("common_voice_dev_verification", "Whispered Common Voice 17", "development_verification", "ceil",
         common, cm_saved, cm_methods, {}),
        ("librispeech_live", "LibriSpeech", "external_live_holdout", "ceil", *lib),
        ("ami_retraining_holdout", "AMI IHM", "retraining_holdout", "ceil", *ami),
        ("ami_audio_aware_holdout", "AMI IHM", "audio_feature_holdout", "floor", *final),
    )
    for filename, dataset, split, rounding, frame, saved, methods, provided in specs:
        frame["avg_logprob"] = pd.to_numeric(frame.avg_logprob.replace("", np.nan))
        frame["true_high_error"] = pd.to_numeric(frame.true_high_error).astype(int)
        if not frame.true_high_error.eq(frame.wer.gt(.20).astype(int)).all():
            raise ValueError(f"WER and high-error labels disagree in {filename}.")
        signals = risk_signals(frame)
        scores = {"classifier": signals.boosted_trees, "native": signals.whisper_confidence,
                  "original_hybrid": blend(signals, .50)}
        for key, (values, _) in provided.items():
            if key in scores:
                np.testing.assert_allclose(values, scores[key], rtol=1e-10, atol=1e-12)
            else:
                scores[key] = values
        table = frame[["speaker_id", "true_high_error", "wer", "transcript", "reference", "avg_logprob"]].copy()
        table.insert(0, "dataset", dataset)
        table.insert(1, "evaluation_split", split)
        table["recording_group"] = frame.get("cluster", frame.get("chapter_id", frame.speaker_id))
        table["native_confidence_missing"] = frame.avg_logprob.isna()
        budget = ceil(len(frame) * .20) if rounding == "ceil" else len(frame) // 5
        total = int(table.true_high_error.sum())
        for name, risk in scores.items():
            selected = table.index.isin(ranked_ids(risk)[:budget])
            caught = int(table.loc[selected, "true_high_error"].sum())
            if name in provided and not np.array_equal(selected, provided[name][1].to_numpy()):
                raise ValueError(f"Saved review membership differs for {filename}/{name}.")
            source_method = methods[name]
            if source_method is not None:
                row = saved.loc[source_method]
                if (budget, caught) != (row.reviewed, row.caught):
                    raise ValueError(f"Saved review counts differ for {filename}/{name}.")
            table[f"{name}_risk"] = risk.replace([np.inf, -np.inf], np.nan)
            table[f"{name}_review"] = selected
            all_metrics.append({"file": filename, "method": name, "clips": len(frame), "high_error": total,
                                "reviewed": budget, "caught": caught, "missed": total - caught,
                                "precision": caught / budget, "recall": caught / total, "budget_rounding": rounding})
        table.to_csv(output / f"{filename}.csv", lineterminator="\n")
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output / "verified_counts.csv", index=False, lineterminator="\n")
    files = (BOOSTED, SPLIT, LIBRISPEECH, AMI_RETRAIN, AMI_AUDIO)
    provenance = {"source_archives_sha256": {name: _file_hash(source / name) for name in files},
                  "english_development_parquet_sha256": _file_hash(development_metadata)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory holding five original Colab result ZIPs")
    parser.add_argument("--development-metadata", required=True, type=Path, help="Pinned Whispered English dev Parquet")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(export(args.source, args.development_metadata, args.output).to_string(index=False))
