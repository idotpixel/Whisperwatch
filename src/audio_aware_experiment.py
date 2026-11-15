"""Isolated, matched audio-feature ablation on fresh AMI meeting groups."""

import hashlib
import io
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from .audio_features import AUDIO_FEATURE_NAMES, extract_audio_features
from .features import FEATURE_NAMES
from .inference import score_features
from .live_retraining import REPO, REVISION, _clusters, _retrieve
from .live_validation import EXPECTED_LIBRARIES, MODEL_MEMBER, MODEL_SHA256, _run_clips
from .second_asr_pilot import diagnostic_wer
from .train import make_boosted_classifier
from .whisper_compatibility import _features


SEED = 20261115
METHODS = ("native_confidence", "original_hybrid", "control", "audio_aware", "control_hybrid", "audio_hybrid")
CONTRASTS = (("audio_aware", "control"), ("audio_hybrid", "control_hybrid"),
             ("audio_hybrid", "native_confidence"), ("audio_hybrid", "original_hybrid"))
GROUP_COUNTS = {"train": 10, "tune": 4, "evaluation": 16}
MIN_EVALUATION_GROUPS = 12
CLIPS_PER_GROUP = {"train": 100, "tune": 25, "evaluation": 40}


def freeze_sample(metadata, prior, shard_tree, output):
    """Draw whole connected groups before seeing audio or scores, then sample clips."""
    frame = metadata.dropna(subset=["audio_id", "meeting_id", "speaker_id", "begin_time", "end_time"]).copy()
    if not frame.audio_id.is_unique:
        raise ValueError("AMI metadata contains duplicate audio IDs.")
    frame["meeting_family"] = frame.meeting_id.str.replace(r"(?<=\d)[a-z]$", "", regex=True)
    frame["cluster"] = _clusters(frame)
    frame["clip_id"] = REPO + "/" + frame.audio_id
    def touched(previous):
        overlap = (frame.speaker_id.isin(previous.speaker_id) | frame.meeting_family.isin(previous.meeting_family)
                   | frame.clip_id.isin(previous.clip_id))
        return set(frame.loc[overlap, "cluster"])

    seen = touched(prior)
    blocked = touched(prior.loc[prior.split.eq("held_out")])
    eligible = frame.loc[~frame.cluster.isin(blocked) & (frame.end_time - frame.begin_time).between(2, 25)]
    # Each group draws from the shard with the most eligible clips so Colab only
    # fetches one shard per group. This rule depends on identifiers/duration only.
    by_shard = eligible.groupby(["cluster", "shard"]).size().rename("clips").reset_index()
    dominant = by_shard.sort_values(["cluster", "clips", "shard"], ascending=[True, False, True])
    dominant = dominant.drop_duplicates("cluster").set_index("cluster")
    rng = np.random.default_rng(SEED)
    train_groups = set(dominant.index[dominant.clips.ge(CLIPS_PER_GROUP["train"])])
    fresh_groups = sorted(set(dominant.index[dominant.clips.ge(CLIPS_PER_GROUP["evaluation"])]) - seen)
    # Prefer previously used training groups so fresh groups are available for evaluation.
    train_order = list(rng.permutation(sorted(train_groups - set(fresh_groups))))
    train_order += list(rng.permutation(sorted(train_groups & set(fresh_groups))))
    train = train_order[:GROUP_COUNTS["train"]]
    available = sorted(set(fresh_groups) - set(train))
    evaluation_count = min(GROUP_COUNTS["evaluation"], len(available) - GROUP_COUNTS["tune"])
    if evaluation_count < MIN_EVALUATION_GROUPS or len(train_groups) < GROUP_COUNTS["train"]:
        raise ValueError(f"Only {len(fresh_groups)} fresh groups and {len(train_groups)} training groups available. "
                         "The frozen design requires at least 12 evaluation groups, four tuning groups, and ten training groups.")
    holdout = list(rng.permutation(available)[:evaluation_count + GROUP_COUNTS["tune"]])
    partitions = {"train": train, "tune": holdout[evaluation_count:], "evaluation": holdout[:evaluation_count]}
    rows = []
    for split, cluster_ids in partitions.items():
        for cluster in cluster_ids:
            candidates = eligible.loc[eligible.cluster.eq(cluster) & eligible.shard.eq(dominant.at[cluster, "shard"])]
            picks = rng.choice(len(candidates), size=CLIPS_PER_GROUP[split], replace=False)
            rows.append(candidates.sort_values("clip_id").iloc[sorted(picks)].assign(split=split))
    manifest = pd.concat(rows).sort_values("clip_id").reset_index(drop=True)
    _check_partitions(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.csv"
    manifest.to_csv(path, index=False, lineterminator="\n")
    shards = {item["path"]: {"bytes": item["size"], "sha256": item["lfs"]["oid"]}
              for item in shard_tree if item["path"] in set(manifest.shard)}
    if set(shards) != set(manifest.shard):
        raise ValueError("Missing shard provenance.")
    plan = {"dataset": REPO, "revision": REVISION, "license": "CC BY 4.0", "seed": SEED,
            "source_rows_checked": len(frame), "excluded_prior_holdout_groups": len(blocked),
            "eligible_fresh_groups": len(fresh_groups), "prior_groups_reused_for_training": len(set(train) & seen),
            "partitions": manifest.groupby("split").agg(clips=("clip_id", "size"),
                speakers=("speaker_id", "nunique"), meetings=("meeting_id", "nunique"),
                independent_groups=("cluster", "nunique")).to_dict("index"),
            "audio_seconds": float((manifest.end_time - manifest.begin_time).sum()),
            "shards": shards, "audio_shard_bytes": sum(item["bytes"] for item in shards.values()),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "preflight.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def _check_partitions(manifest):
    for column in ("speaker_id", "meeting_id", "meeting_family", "cluster"):
        if manifest.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Split leakage through {column}.")
    for split, groups in GROUP_COUNTS.items():
        data = manifest.loc[manifest.split.eq(split)]
        if split == "evaluation":
            groups = data.cluster.nunique()
            if not MIN_EVALUATION_GROUPS <= groups <= GROUP_COUNTS["evaluation"]:
                raise ValueError("Unexpected number of evaluation groups.")
        if len(data) != groups * CLIPS_PER_GROUP[split] or data.cluster.nunique() != groups:
            raise ValueError(f"Unexpected sample count for {split}.")


def _fit_matched(features, target, names):
    if target.nunique() != 2:
        raise ValueError("Both high-error and low-error training clips are needed.")
    model = make_boosted_classifier()
    model.fit(features.loc[target.index, list(names)], target.astype(int))
    return {"pipeline": model, "feature_names": names, "sklearn_version": version("scikit-learn"),
            "training_rows": len(target), "use": "Experimental batch ranking"}


def _scores(native, original, control, audio):
    signals = pd.DataFrame({"native_confidence": native, "original": original, "control": control, "audio_aware": audio})
    rank = signals.rank(method="average", pct=True)
    for model, hybrid in (("original", "original_hybrid"), ("control", "control_hybrid"), ("audio_aware", "audio_hybrid")):
        signals[hybrid] = (rank["native_confidence"] + rank[model]) / 2
    return signals[list(METHODS)]


def _review(scores, labels):
    # Never exceed the budget if a failure leaves a count indivisible by five.
    budget = len(labels) // 5
    picked = pd.DataFrame(False, index=scores.index, columns=METHODS)
    for name in METHODS:
        ordered = scores[name].sort_index().sort_values(ascending=False, kind="stable").index
        picked.loc[ordered[:budget], name] = True
    true = picked.loc[labels]
    hits = true.sum()
    return picked, pd.DataFrame({"clips": len(labels), "high_error": int(labels.sum()), "reviewed": budget,
        "caught": hits, "missed": labels.sum() - hits, "precision": hits / (budget or np.nan),
        "recall": hits / (labels.sum() or np.nan), "delta_vs_native": hits - hits.native_confidence,
        "unique_vs_native": (true & ~true.native_confidence.to_numpy()[:, None]).sum(),
        "lost_vs_native": (~true & true.native_confidence.to_numpy()[:, None]).sum()})


def _resample(native, original, control, audio, labels, clusters, draws=2000):
    groups = list(clusters.groupby(clusters, sort=True).indices.values())
    result = pd.DataFrame(columns=["method", "baseline", "unit", "lower_95", "upper_95", "draws"])
    if len(groups) < 2 or not labels.any():
        return result
    rng = np.random.default_rng(SEED)
    values = [series.to_numpy() for series in (native, original, control, audio)]
    target = labels.to_numpy()
    outcomes = []
    for _ in range(draws):
        # Rows are sorted by clip ID. Stable sort breaks ties by that ID and
        # then the occurrence number when the same group is drawn twice.
        take = np.sort(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
        budget = len(take) // 5
        y = target[take]
        score = _scores(*(array[take] for array in values))
        hits = {name: int(y[np.argsort(-score[name].to_numpy(), kind="stable")[:budget]].sum()) for name in METHODS}
        delta = np.array([hits[name] - hits[baseline] for name, baseline in CONTRASTS])
        outcomes.append(np.stack([delta, delta / (budget or np.nan), delta / (y.sum() or np.nan)]))
    data = np.stack(outcomes)
    records = []
    for index, (name, baseline) in enumerate(CONTRASTS):
        for unit_index, unit in enumerate(("detections", "precision", "recall")):
            sample = data[:, unit_index, index]
            sample = sample[np.isfinite(sample)]
            low, high = np.quantile(sample, [.025, .975]) if len(sample) else (np.nan, np.nan)
            records.append((name, baseline, unit, low, high, len(sample)))
    return pd.DataFrame(records, columns=result.columns)


def run(plan_dir, boosted_zip, output):
    """Decode and train in hosted Colab, then export scores from the frozen holdout."""
    if not Path("/content").is_dir() or "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Use a hosted Google Colab GPU runtime.")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    for library, expected in {**EXPECTED_LIBRARIES, "scikit-learn": "1.8.0"}.items():
        if version(library) != expected:
            raise ValueError(f"Use {library}=={expected} for this experiment.")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 6_000_000_000:
        raise OSError("Allow at least 6 GB of free disk after installing dependencies.")
    sample = (plan_dir / "manifest.csv").read_bytes()
    plan = json.loads((plan_dir / "preflight.json").read_text(encoding="utf-8"))
    if plan["revision"] != REVISION or hashlib.sha256(sample).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("The plan and frozen manifest disagree.")
    manifest = pd.read_csv(io.BytesIO(sample)).set_index("clip_id", verify_integrity=True).sort_index()
    _check_partitions(manifest)
    (output / "manifest.csv").write_bytes(sample)
    with ZipFile(boosted_zip) as bundle:
        original_bytes = bundle.read(MODEL_MEMBER)
    if hashlib.sha256(original_bytes).hexdigest() != MODEL_SHA256:
        raise ValueError("Upload the original frozen boosted tree results ZIP.")
    original = joblib.load(io.BytesIO(original_bytes))

    retrieval = _retrieve(manifest.reset_index(), plan, output)
    if not retrieval.status.eq("ok").any():
        raise RuntimeError("No audio recovered. See retrieval.csv.")
    print(f"Audio retrieval complete for {retrieval.status.eq('ok').sum()} clips. Loading Whisper.", flush=True)
    settings, live = _run_clips(retrieval, output)
    failure = live.get("error", pd.Series(index=live.index, dtype=str))
    duration = live.get("duration_s", pd.Series(index=live.index, dtype=float))
    valid_duration = duration.between(0, 30, inclusive="right")
    status = retrieval[["status"]].rename(columns={"status": "retrieval_status"})
    status["transcription_status"] = "not_run"
    status.loc[live.index, "transcription_status"] = np.where(failure.notna(), "failed",
                                                              np.where(valid_duration, "ok", "unsupported_duration"))
    status["transcription_error"] = failure
    status = status.join(manifest[["split", "cluster"]])
    status.to_csv(output / "clip_status.csv")
    live = live.loc[failure.isna() & valid_duration]
    if live.empty or set(manifest.loc[live.index, "split"]) != set(GROUP_COUNTS):
        raise RuntimeError("Each partition needs successful transcriptions. See clip_status.csv.")

    print("Measuring the four audio features.", flush=True)
    waveform_rows = []
    for clip_id, path in retrieval.loc[live.index, "audio_path"].items():
        started = perf_counter()
        features = extract_audio_features(path)
        waveform_rows.append({"clip_id": clip_id, **features, "audio_feature_seconds": perf_counter() - started})
    waveform = pd.DataFrame(waveform_rows).set_index("clip_id")
    audio_missing = waveform[list(AUDIO_FEATURE_NAMES)].isna().any(axis=1)
    status["audio_feature_status"] = audio_missing.map({True: "missing", False: "ok"}).reindex(status.index).fillna("not_run")
    status.to_csv(output / "clip_status.csv")
    features = _features(live).join(waveform[list(AUDIO_FEATURE_NAMES)])
    metadata = manifest.loc[live.index, ["split", "speaker_id", "meeting_id", "meeting_family", "cluster"]]

    def labels(index):
        return pd.Series([diagnostic_wer(ref, text) for ref, text in
                         zip(retrieval.loc[index, "reference"], live.loc[index, "transcript"])], index=index, name="wer")

    train_ids = metadata.index[metadata.split.eq("train")]
    train_y = labels(train_ids).gt(.20)
    models = {}
    for name, schema in (("control", FEATURE_NAMES), ("audio_aware", FEATURE_NAMES + AUDIO_FEATURE_NAMES)):
        artifact = _fit_matched(features, train_y, schema)
        artifact_path = output / f"ami_{name}_experiment.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        models[name] = joblib.load(artifact_path)

    # Tune is reserved for checking coverage. Both models are fixed a priori.
    evaluated = metadata.loc[metadata.split.eq("evaluation")]
    idx = evaluated.index
    predictors = features.loc[idx].to_dict("records")
    raw_scores = [-features.loc[idx, "avg_logprob"].fillna(-np.inf),
                  *[pd.Series(score_features(artifact, predictors), index=idx)
                    for artifact in (original, models["control"], models["audio_aware"])]]
    scores = _scores(*raw_scores)
    wer = labels(idx)
    y = wer.gt(.20)
    reviewed, performance = _review(scores, y)
    performance = performance.assign(speakers=evaluated.speaker_id.nunique(), groups=evaluated.cluster.nunique())
    performance.to_csv(output / "comparison.csv", index_label="method")
    intervals = _resample(*raw_scores, y, evaluated.cluster)
    intervals.to_csv(output / "paired_bootstrap.csv", index=False)
    deltas = []
    for group in sorted(evaluated.cluster.unique()):
        keep = evaluated.cluster.ne(group)
        if keep.any():
            group_scores = _scores(*(signal.loc[keep] for signal in raw_scores))
            _, table = _review(group_scores, y.loc[keep])
            for method, baseline in CONTRASTS:
                deltas.append({"omitted_group": group, "method": method, "baseline": baseline,
                    "detection_change": table.at[method, "caught"] - table.at[baseline, "caught"],
                    "recall_change": table.at[method, "recall"] - table.at[baseline, "recall"]})
    pd.DataFrame(deltas).to_csv(output / "leave_one_group_out.csv", index=False)
    audit = evaluated.join(features).join(scores.add_prefix("score_")).join(reviewed.add_prefix("review_"))
    audit = audit.assign(wer=wer, high_error=y, transcript=live.transcript, reference=retrieval.reference)
    audit.to_csv(output / "evaluation_predictions.csv")
    examples = []
    for method, baseline in CONTRASTS:
        for left, right, direction in ((method, baseline, "gained"), (baseline, method, "lost")):
            cases = audit.loc[y & reviewed[left] & ~reviewed[right]].sort_values("wer", ascending=False)
            examples.append(cases.assign(method=method, baseline=baseline, direction=direction))
    pd.concat(examples).to_csv(output / "changed_detections.csv")
    waveform.to_csv(output / "audio_measurements.csv")
    timings = waveform.audio_feature_seconds
    runtime = {"settings": settings, "original_sha256": MODEL_SHA256, "training_high_error": int(train_y.sum()),
               "counts": status.groupby(["split", "retrieval_status", "transcription_status"]).size().reset_index(name="clips").to_dict("records"),
               "evaluation_groups": int(evaluated.cluster.nunique()), "audio_feature_missing": int(audio_missing.sum()),
               "audio_feature_median_ms": float(1000 * timings.median()), "audio_feature_p95_ms": float(1000 * timings.quantile(.95)),
               "audio_feature_total_seconds": float(timings.sum()), "whisper_seconds": float(live.seconds.sum())}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    report = ["# Audio-aware review experiment", "",
              f"Each ranking reviewed {len(y) // 5} of {len(y)} eligible clips across "
              f"{runtime['evaluation_groups']} held-out meeting groups. {int(y.sum())} clips had WER above 20%.", ""]
    for method, row in performance.iterrows():
        report.append(f"- {method} caught {int(row.caught)} high-error clips with {row.precision:.1%} precision "
                      f"and {row.recall:.1%} recall.")
    report.extend(["", f"Extra audio processing took a median of {runtime['audio_feature_median_ms']:.2f} ms "
                   f"per measured clip ({runtime['audio_feature_p95_ms']:.2f} ms at the 95th percentile).", ""])
    for row in intervals.loc[intervals.unit.eq("recall")].itertuples():
        delta = performance.at[row.method, "recall"] - performance.at[row.baseline, "recall"]
        report.append(f"- {row.method} versus {row.baseline}: {100 * delta:+.1f} recall points "
                      f"(paired 95% interval {100 * row.lower_95:+.1f} to {100 * row.upper_95:+.1f}).")
    report.extend(["", "Interpret the audio versus control contrasts alongside the comparison with native confidence. "
                   "This holdout represents AMI headset segments. The number and diversity of meeting groups limit "
                   "the strength of conclusions about subtitle audio outside AMI."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return performance, intervals, runtime
