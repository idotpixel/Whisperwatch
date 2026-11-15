"""Train one candidate on live AMI features and evaluate once on held-out meetings."""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .ensemble_experiment import blend
from .features import FEATURE_NAMES
from .inference import MODEL_FORMAT, load_model, score_features
from .live_validation import (BOOTSTRAP_SEED, EXPECTED_LIBRARIES, MODEL_MEMBER, MODEL_SHA256, RangeReader,
                              _bootstrap, _metrics, _rank, _run_clips, dataset_url)
from .second_asr_pilot import diagnostic_wer, tokens
from .train import make_boosted_classifier
from .whisper_compatibility import _features


REPO = "edinburghcstr/ami"
REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"
SAMPLE_SEED = 20261114
COUNTS = {"train": 1500, "held_out": 750}


def _source_metadata(item):
    reader = RangeReader(dataset_url(REPO, REVISION, item["path"]), item["size"], 8_000_000)
    columns = ["audio_id", "meeting_id", "speaker_id", "begin_time", "end_time"]
    data = pq.ParquetFile(reader).read(columns=columns, use_threads=False).to_pandas()
    split = "train" if "/train-" in item["path"] else "held_out"
    return data.assign(split=split, shard=item["path"]), reader.transferred


def _split_sample(frame):
    frame = frame.dropna(subset=["audio_id", "meeting_id", "speaker_id", "begin_time", "end_time"]).copy()
    # The a/b/c/d meetings from a scenario are recorded with the same team.
    frame["meeting_family"] = frame.meeting_id.map(lambda value: re.sub(r"(?<=\d)[a-z]$", "", value))
    frame["clip_id"] = REPO + "/" + frame.audio_id
    if not frame.clip_id.is_unique:
        raise ValueError("Duplicate source clip IDs in selected shards.")
    held_out = frame.loc[frame.split.eq("held_out")]
    training = frame.loc[frame.split.eq("train")]
    in_test = (training.speaker_id.isin(held_out.speaker_id)
               | training.meeting_id.isin(held_out.meeting_id)
               | training.meeting_family.isin(held_out.meeting_family))
    eligible = (frame.end_time - frame.begin_time).between(2, 25)
    pools = {"train": training.loc[~in_test & eligible], "held_out": held_out.loc[eligible]}
    selected = []
    for split, pool in pools.items():
        if len(pool) < COUNTS[split]:
            raise ValueError(f"Too few eligible {split} clips after the separation checks.")
        selected.append(pool.sort_values("clip_id").sample(n=COUNTS[split], random_state=SAMPLE_SEED))
    manifest = pd.concat(selected).sort_values("clip_id").reset_index(drop=True)
    for key in ("speaker_id", "meeting_id", "meeting_family"):
        if set(manifest.loc[manifest.split.eq("train"), key]) & set(held_out[key]):
            raise ValueError(f"Held-out overlap in {key}.")
    return manifest, int(in_test.sum()), {split: len(pool) for split, pool in pools.items()}


def prepare(output):
    """Freeze using only metadata. This function never reads a reference or an audio column."""
    path = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/ihm"
    with urlopen(path, timeout=60) as response:
        tree = json.load(response)
    training = sorted((item for item in tree if "/train-" in item["path"]), key=lambda item: item["path"])[:6]
    held_out = sorted((item for item in tree if "/test-" in item["path"]), key=lambda item: item["path"])
    if len(training) != 6 or len(held_out) != 4:
        raise ValueError("Unexpected AMI headset shard layout.")
    files = training + held_out
    with ThreadPoolExecutor(max_workers=6) as pool:
        metadata = list(pool.map(_source_metadata, files))
    manifest, excluded, eligible = _split_sample(pd.concat([data for data, _ in metadata], ignore_index=True))
    output.mkdir(parents=True, exist_ok=True)
    file = output / "manifest.csv"
    manifest.to_csv(file, index=False, lineterminator="\n")
    shards = {item["path"]: {"bytes": item["size"], "sha256": item["lfs"]["oid"]} for item in files
              if item["path"] in set(manifest.shard)}
    details = {"dataset": REPO, "license": "CC BY 4.0", "revision": REVISION, "seed": SAMPLE_SEED,
               "metadata_bytes": sum(count for _, count in metadata), "eligible": eligible,
               "training_rows_removed_for_holdout_overlap": excluded, "shards": shards,
               "audio_shard_bytes": sum(item["bytes"] for item in shards.values()),
               "manifest_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
               "partitions": manifest.groupby("split").agg(clips=("clip_id", "size"), speakers=("speaker_id", "nunique"),
                                                           meetings=("meeting_id", "nunique"),
                                                           families=("meeting_family", "nunique")).to_dict("index")}
    details["sample_duration_seconds"] = float((manifest.end_time - manifest.begin_time).sum())
    details["held_out_bootstrap_clusters"] = int(_clusters(manifest.loc[manifest.split.eq("held_out")]).nunique())
    (output / "preflight.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def _download_shard(shard, metadata, target):
    if not target.is_file():
        partial = target.with_suffix(".part")
        try:
            with urlopen(dataset_url(REPO, REVISION, shard), timeout=120) as source, partial.open("wb") as dest:
                total = 0
                for block in iter(lambda: source.read(1 << 20), b""):
                    total += len(block)
                    if total > metadata["bytes"]:
                        raise OSError("The shard exceeded its pinned size.")
                    dest.write(block)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    with target.open("rb") as source:
        valid = target.stat().st_size == metadata["bytes"] and hashlib.file_digest(source, "sha256").hexdigest() == metadata["sha256"]
    if not valid:
        target.unlink()
        raise OSError(f"SHA-256 or size check failed for {shard}.")


def _retrieve(manifest, plan, output):
    folder = output / "audio"
    folder.mkdir(exist_ok=True)
    rows = []
    for shard, group in manifest.groupby("shard"):
        print(f"Reading AMI {shard}", flush=True)
        path = output / Path(shard).name
        wanted = group.set_index("audio_id")
        found = set()
        try:
            _download_shard(shard, plan["shards"][shard], path)
            fields = ["audio_id", "speaker_id", "meeting_id", "audio", "text"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=24, columns=fields):
                for item in batch.to_pylist():
                    key = item["audio_id"]
                    if key not in wanted.index:
                        continue
                    clip = wanted.loc[key]
                    if key in found or (clip.speaker_id, clip.meeting_id) != (item["speaker_id"], item["meeting_id"]):
                        raise ValueError(f"AMI recording provenance changed for {key}.")
                    found.add(key)
                    audio = item["audio"]["bytes"]
                    if not audio or not tokens(item["text"] or ""):
                        rows.append({"clip_id": clip.clip_id, "status": "missing_audio_or_reference"})
                        continue
                    location = folder / (hashlib.sha256(key.encode()).hexdigest() + ".audio")
                    location.write_bytes(audio)
                    rows.append({"clip_id": clip.clip_id, "status": "ok", "audio_path": str(location),
                                 "hash": hashlib.sha256(audio).hexdigest(), "reference": item["text"]})
        except OSError as error:
            rows.extend({"clip_id": wanted.loc[key, "clip_id"], "status": "shard_failed", "error": str(error)}
                        for key in wanted.index.difference(found))
            found = set(wanted.index)
        finally:
            path.unlink(missing_ok=True)
        rows.extend({"clip_id": wanted.loc[key, "clip_id"], "status": "missing_clip"}
                    for key in wanted.index.difference(found))
    frame = pd.DataFrame(rows).set_index("clip_id", verify_integrity=True).sort_index()
    if frame.status.eq("ok").any():
        repeats = frame.loc[frame.status.eq("ok"), "hash"].duplicated(keep=False)
        frame.loc[repeats[repeats].index, "status"] = "duplicate_audio_bytes"
    frame.drop(columns="reference", errors="ignore").to_csv(output / "retrieval.csv")
    return frame


def _fit_candidate(features, labels):
    if labels.nunique() != 2:
        raise ValueError("The training partition needs both classes.")
    model = make_boosted_classifier()
    model.fit(features.loc[labels.index, list(FEATURE_NAMES)], labels.astype(int))
    return {"format_version": MODEL_FORMAT, "feature_names": FEATURE_NAMES, "pipeline": model,
            "training_source": REPO, "training_revision": REVISION, "training_rows": len(labels),
            "wer_threshold": .20, "sklearn_version": version("scikit-learn"),
            "usage": "Experimental batch ranking at a 20% review budget"}


def _evaluate(features, labels, metadata, candidate, original, output):
    index = labels.index
    fresh = _rank(features, score_features(candidate, features.to_dict("records")))
    old = _rank(features, score_features(original, features.to_dict("records")))
    exports = metadata.copy()
    tables = []
    intervals = []
    for name, scores in (("archived_training", old), ("live_training", fresh)):
        reviewed, metrics = _metrics(scores, labels)
        exports = exports.join(scores.add_prefix(f"{name}_score_")).join(reviewed.add_prefix(f"{name}_review_"))
        tables.append(metrics.assign(classifier=name).rename_axis("method").reset_index())
        intervals.append(_bootstrap(scores, labels, metadata.cluster).assign(classifier=name))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output / "comparison.csv", index=False)
    uncertainty = pd.concat(intervals, ignore_index=True)
    uncertainty = pd.concat([uncertainty, _paired_hybrid_change(old, fresh, labels, metadata.cluster)], ignore_index=True)
    uncertainty.to_csv(output / "paired_bootstrap.csv", index=False)
    sensitivity = []
    if metadata.cluster.nunique() > 1:
        for cluster in sorted(metadata.cluster.unique()):
            keep = metadata.cluster.ne(cluster)
            subset = features.loc[keep]
            rescored = _rank(subset, fresh.loc[keep, "boosted_trees"])
            _, result = _metrics(rescored, labels.loc[keep])
            native = result.loc["native_confidence"]
            hybrid = result.loc["hybrid_50_50"]
            sensitivity.append({"omitted_group": cluster, "remaining_clips": int(hybrid.evaluated),
                                "high_error_total": int(hybrid.high_error_total),
                                "extra_caught_vs_native": int(hybrid.caught - native.caught),
                                "recall_change": hybrid.recall - native.recall})
    pd.DataFrame(sensitivity).to_csv(output / "leave_one_meeting_group_out.csv", index=False)
    exports = exports.join(features).assign(high_error=labels)
    return table, uncertainty, exports.loc[index]


def _paired_hybrid_change(old, fresh, labels, clusters, draws=2000):
    groups = list(clusters.groupby(clusters, sort=True).indices.values())
    rows = []
    if len(groups) < 2 or not labels.any():
        return pd.DataFrame()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y = labels.to_numpy()
    confidence = old.native_confidence.to_numpy()
    trees = [scores.boosted_trees.to_numpy() for scores in (old, fresh)]
    for _ in range(draws):
        indices = np.sort(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
        budget = (len(indices) + 4) // 5
        target = y[indices]
        detections = []
        for tree in trees:
            signals = pd.DataFrame({"boosted_trees": tree[indices], "whisper_confidence": confidence[indices]})
            priority = blend(signals, .50).to_numpy()
            detections.append(int(target[np.argsort(-priority, kind="stable")[:budget]].sum()))
        delta = detections[1] - detections[0]
        rows.append((delta, delta / budget, delta / (target.sum() or np.nan)))
    result = []
    for unit, values in zip(("detections", "precision", "recall"), np.array(rows).T):
        kept = values[np.isfinite(values)]
        low, high = np.quantile(kept, [.025, .975]) if len(kept) else (np.nan, np.nan)
        result.append({"classifier": "live_training", "method": "hybrid_vs_archived_hybrid", "unit": unit,
                       "lower_95": low, "upper_95": high, "draws": len(kept)})
    return pd.DataFrame(result)


def _clusters(metadata):
    """Joined meeting families and shared speakers are indivisible bootstrap units."""
    links = {}
    for family, group in metadata.groupby("meeting_family"):
        links[family] = set(group.speaker_id)
    clusters = {}
    while links:
        seed = next(iter(links))
        speakers = links.pop(seed)
        component = {seed}
        while True:
            neighbors = [family for family, ids in links.items() if ids & speakers]
            if not neighbors:
                break
            for family in neighbors:
                speakers |= links.pop(family)
                component.add(family)
        clusters.update(dict.fromkeys(component, min(component)))
    return metadata.meeting_family.map(clusters)


def run(plan_dir, boosted_zip, output):
    if not Path("/content").is_dir() or "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("This experiment runs in hosted Google Colab.")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    for name, expected in EXPECTED_LIBRARIES.items():
        if version(name) != expected:
            raise ValueError(f"Use {name}=={expected} for the frozen decoder.")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 6_000_000_000:
        raise OSError("Allow at least 6 GB of free runtime disk after dependency setup.")
    plan = json.loads((plan_dir / "preflight.json").read_text())
    locked = (plan_dir / "manifest.csv").read_bytes()
    if plan["revision"] != REVISION or hashlib.sha256(locked).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("The sample differs from the frozen AMI manifest.")
    manifest = pd.read_csv(io.BytesIO(locked)).set_index("clip_id", verify_integrity=True).sort_index()
    if manifest.split.value_counts().to_dict() != COUNTS:
        raise ValueError("Incorrect frozen train and held-out counts.")
    for key in ("speaker_id", "meeting_id", "meeting_family"):
        if set(manifest.loc[manifest.split.eq("train"), key]) & set(manifest.loc[manifest.split.eq("held_out"), key]):
            raise ValueError(f"Partition overlap in {key}.")
    (output / "manifest.csv").write_bytes(locked)
    with ZipFile(boosted_zip) as bundle:
        model = bundle.read(MODEL_MEMBER)
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError("Use the original frozen boosted results ZIP.")
    source_model = output / "archived_classifier.tmp"
    source_model.write_bytes(model)
    try:
        original = load_model(source_model)
    finally:
        source_model.unlink()
    retrieval = _retrieve(manifest.reset_index(), plan, output)
    if not retrieval.status.eq("ok").any():
        raise RuntimeError("No audio recovered. Check retrieval.csv.")
    settings, live = _run_clips(retrieval, output)
    failure = live.get("error", pd.Series(index=live.index, dtype=str))
    supported = live.get("duration_s", pd.Series(index=live.index, dtype=float)).between(0, 30, inclusive="right")
    status = retrieval[["status"]].rename(columns={"status": "retrieval_status"})
    status["transcription_status"] = "not_run"
    status.loc[live.index, "transcription_status"] = np.where(failure.notna(), "failed", np.where(supported, "ok", "unsupported_duration"))
    status["error"] = failure
    status.join(manifest[["split"]]).to_csv(output / "clip_status.csv")
    live = live.loc[failure.isna() & supported]
    metadata = manifest.loc[live.index, ["split", "speaker_id", "meeting_id", "meeting_family"]]
    training = metadata.index[metadata.split.eq("train")]
    held_out = metadata.index[metadata.split.eq("held_out")]
    if training.empty or held_out.empty:
        raise RuntimeError("Both partitions need evaluated clips. Check clip_status.csv.")
    features = _features(live)

    def evaluation_targets(index):
        return pd.Series([diagnostic_wer(ref, text) for ref, text in
                          zip(retrieval.loc[index, "reference"], live.loc[index, "transcript"])], index=index)

    train_wer = evaluation_targets(training)
    candidate = _fit_candidate(features, train_wer.gt(.20))
    model_file = output / "ami_live_candidate.joblib"
    joblib.dump(candidate, model_file, compress=3)
    candidate = load_model(model_file)
    held_wer = evaluation_targets(held_out)
    held_metadata = metadata.loc[held_out].copy()
    held_metadata["cluster"] = _clusters(manifest.loc[manifest.split.eq("held_out")]).loc[held_out]
    metrics, uncertainty, exports = _evaluate(features.loc[held_out], held_wer.gt(.20), held_metadata,
                                              candidate, original, output)
    predictions = exports.assign(wer=held_wer, reference=retrieval.reference, transcript=live.transcript)
    predictions.to_csv(output / "held_out_predictions.csv")
    hybrid = predictions.live_training_review_hybrid_50_50
    native = predictions.live_training_review_native_confidence
    for name, selected in (("live_hybrid_only", hybrid & ~native), ("native_confidence_only", native & ~hybrid)):
        predictions.loc[predictions.high_error & selected].sort_values("wer", ascending=False).to_csv(output / f"{name}.csv")
    runtime = {"settings": settings, "seed": SAMPLE_SEED, "train_scored": len(training),
               "train_high_error": int(train_wer.gt(.20).sum()), "held_out_scored": len(held_out),
               "held_out_high_error": int(held_wer.gt(.20).sum()),
               "held_out_bootstrap_clusters": int(held_metadata.cluster.nunique()),
               "holdout_exclusion_rate": 1 - len(held_out) / COUNTS["held_out"],
               "original_sha256": MODEL_SHA256,
               "candidate_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest()}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    comparison = metrics.set_index(["classifier", "method"])
    native = comparison.loc[("live_training", "native_confidence")]
    new = comparison.loc[("live_training", "hybrid_50_50")]
    old = comparison.loc[("archived_training", "hybrid_50_50")]
    interval = uncertainty.loc[uncertainty.classifier.eq("live_training")
                               & uncertainty.method.eq("hybrid_50_50") & uncertainty.unit.eq("recall")]
    report = ["# Live-data retraining on AMI meetings", "",
              f"At {int(native.reviewed)} review slots, native confidence caught {int(native.caught)} errors, "
              f"the original hybrid caught {int(old.caught)}, and the retrained hybrid caught {int(new.caught)} "
              f"of {runtime['held_out_high_error']} high-error clips."]
    if not interval.empty:
        bounds = interval.iloc[0]
        report.extend(["", f"The retrained hybrid's recall difference versus native confidence was "
                             f"{100 * (new.recall - native.recall):+.1f} percentage points. "
                             f"The paired bootstrap interval was {100 * bounds.lower_95:+.1f} to "
                             f"{100 * bounds.upper_95:+.1f} percentage points."])
    report.extend(["", f"The holdout had {runtime['held_out_bootstrap_clusters']} disjoint meeting groups "
                  "for resampling. A small number of groups limits how well the interval represents future meetings. "
                  "See the leave-one-group-out table for a check on whether a single group drives the result. "
                  "The comparison applies to sampled AMI headset utterances."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics, uncertainty, runtime
