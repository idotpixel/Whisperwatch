"""Frozen external validation. The preflight reads metadata columns by byte range."""

import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .ensemble_experiment import blend, review_queue, risk_signals
from .features import FEATURE_NAMES
from .inference import load_model, score_features
from .second_asr_pilot import diagnostic_wer, tokens
from .train import DATASET, FULL_TRAIN_SHARDS, REVISION
from .whisper_compatibility import MODEL_MEMBER, _features


SOURCE = "openslr/librispeech_asr"
SOURCE_REVISION = "8edc81a625bfc3d004c8a5d3870e0ade8e1cc59c"
SHARDS = {"clean": (350452636, "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0"),
          "other": (332873172, "38e0c86a8104585c577badd707ca4331e20fa2c645f46180af0b7bcdecff9249")}
SAMPLE_SEED = 20261114
RANDOM_SEED = 20261115
BOOTSTRAP_SEED = 20261116
MODEL_SHA256 = "8d197a7d191a63db1fbbdf31bceaf0badea2b07a88b8df756ccd7283bb6dc496"
EXPECTED_LIBRARIES = {"faster-whisper": "1.2.1", "ctranslate2": "4.9.0", "av": "16.0.1"}
METHODS = ("random", "native_confidence", "boosted_trees", "hybrid_50_50")


def dataset_url(repo, revision, file):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}?download=true"


class RangeReader(io.RawIOBase):
    """Accept only HTTP 206 responses so preflight cannot silently fetch a full shard."""

    def __init__(self, url, size, budget):
        self.url, self.size, self.budget = url, size, budget
        self.offset = self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = {0: 0, 1: self.offset, 2: self.size}[whence] + offset
        return self.offset

    def read(self, size=-1):
        remaining = max(self.size - self.offset, 0)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if self.transferred + size > self.budget:
            raise ValueError("Parquet metadata exceeded its byte-range budget.")
        start, end = self.offset, self.offset + size - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise OSError("The server did not honor the exact metadata byte range.")
            data = response.read(size + 1)
        if len(data) != size:
            raise OSError("Incomplete metadata byte range.")
        self.offset += size
        self.transferred += size
        return data


def freeze_sample(output):
    """Choose 500 IDs uniformly from the sorted two-shard metadata before scoring."""
    frames = []
    transferred = 0
    for group, (size, _) in SHARDS.items():
        relative = f"{group}/test/0000.parquet"
        reader = RangeReader(dataset_url(SOURCE, SOURCE_REVISION, relative), size, 1_000_000)
        frame = pq.ParquetFile(reader).read(columns=["file", "id", "speaker_id", "chapter_id"],
                                           use_threads=False).to_pandas()
        transferred += reader.transferred
        frame = frame.rename(columns={"id": "source_clip_id"}).assign(group=group, shard=relative)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["filename"] = frame.pop("file").map(lambda value: Path(value).name)
    frame["clip_id"] = SOURCE + "/" + frame.source_clip_id
    if (len(frame) != 5559 or not frame.clip_id.is_unique or not frame.filename.is_unique
            or not frame.source_clip_id.str.fullmatch(r"\d+-\d+-\d+").all()
            or not frame.filename.eq(frame.source_clip_id + ".flac").all()):
        raise ValueError("Unexpected LibriSpeech metadata or duplicate identifiers.")
    frame = frame.sort_values("clip_id").reset_index(drop=True)
    selected = sorted(random.Random(SAMPLE_SEED).sample(range(len(frame)), 500))
    manifest = frame.iloc[selected]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.csv"
    manifest.to_csv(path, index=False, lineterminator="\n")
    audit = {"source": SOURCE, "source_revision": SOURCE_REVISION, "eligible_metadata_rows": len(frame),
             "metadata_range_bytes": transferred, "sample_seed": SAMPLE_SEED,
             "sample_clips": len(manifest), "speakers": int(manifest.speaker_id.nunique()),
             "clips_by_group": manifest.group.value_counts().to_dict(),
             "audio_shard_bytes": sum(size for size, _ in SHARDS.values()),
             "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return manifest, audit


def overlap_audit(manifest):
    """Check identifiers across the pinned Whispered train/dev metadata only."""
    base = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/common_voice_17/en"
    cols = ["clip_id", "source", "source_revision", "speaker_id", "audio_ref", "audio_sha256"]
    keys = ("clip_id", "raw_clip_id", "filename", "filename_stem", "raw_speaker_id")
    candidates = dict(zip(keys, (set(manifest.clip_id), set(manifest.source_clip_id), set(manifest.filename),
                                 set(manifest.filename.map(lambda f: Path(f).stem)), set(manifest.speaker_id.astype(str)))))
    matches = dict.fromkeys(keys, 0)
    rows = transfer = hashes = 0
    sources, revisions, audio_datasets = set(), set(), set()
    tasks = []
    for split, shards in (("train", FULL_TRAIN_SHARDS), ("dev", (0,))):
        with urlopen(f"{base}/{split}", timeout=60) as response:
            entries = {item["path"]: item["size"] for item in json.load(response) if item["type"] == "file"}
        for number in shards:
            path = f"common_voice_17/en/{split}/{split}-{number:05d}.parquet"

            tasks.append((path, entries[path]))

    def read_identifiers(task):
        path, size = task
        reader = RangeReader(dataset_url(DATASET, REVISION, path), size, 6_000_000)
        return pq.ParquetFile(reader).read(columns=cols, use_threads=False).to_pydict(), reader.transferred

    with ThreadPoolExecutor(max_workers=6) as pool:
        for data, bytes_read in pool.map(read_identifiers, tasks):
            transfer += bytes_read
            rows += len(data["clip_id"])
            files = [r["file"] for r in data["audio_ref"]]
            for key, values in zip(keys, (data["clip_id"], [v.rsplit("/", 1)[-1] for v in data["clip_id"]],
                                           files, [Path(v).stem for v in files], data["speaker_id"])):
                matches[key] += sum(str(value) in candidates[key] for value in values)
            hashes += sum(bool(value) for value in data["audio_sha256"])
            sources.update(data["source"])
            revisions.update(data["source_revision"])
            audio_datasets.update(r["dataset"] for r in data["audio_ref"])
    return {"whispered_revision": REVISION, "train_dev_metadata_rows": rows, "id_matches": matches,
            "whispered_sources": sorted(sources), "whispered_source_revisions": sorted(revisions),
            "whispered_audio_datasets": sorted(audio_datasets), "whispered_nonempty_audio_hashes": hashes,
            "whispered_metadata_range_bytes": transfer,
            "limits": "Speaker IDs use different systems. No shared identity map or cross-codec audio fingerprints."
                      " This audit does not establish what Large V3 saw during pretraining."}


def preflight(output):
    manifest, audit = freeze_sample(output)
    audit.update(overlap_audit(manifest))
    (output / "preflight.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _retrieve(manifest, output):
    records, references = [], {}
    folder = output / "audio"
    folder.mkdir(exist_ok=True)
    for group, (size, digest) in SHARDS.items():
        subset = manifest.loc[manifest.group.eq(group)].set_index("source_clip_id")
        selected_ids = set(subset.index)
        seen = set()
        shard = output / f"librispeech-{group}.parquet"
        partial = shard.with_suffix(".part")
        try:
            if not shard.is_file():
                url = dataset_url(SOURCE, SOURCE_REVISION, subset.shard.iloc[0])
                with urlopen(url, timeout=120) as response, partial.open("wb") as target:
                    total = 0
                    for block in iter(lambda: response.read(1 << 20), b""):
                        total += len(block)
                        if total > size:
                            raise OSError("Audio shard exceeded its pinned byte size.")
                        target.write(block)
                partial.replace(shard)
            with shard.open("rb") as source:
                valid_shard = shard.stat().st_size == size and hashlib.file_digest(source, "sha256").hexdigest() == digest
            if not valid_shard:
                shard.unlink(missing_ok=True)
                raise OSError("Audio shard failed its pinned size or SHA-256 check.")
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["id", "audio", "text", "speaker_id"]):
                for item in batch.to_pylist():
                    key = item["id"]
                    if key not in selected_ids:
                        continue
                    row = subset.loc[key]
                    audio = item["audio"]["bytes"]
                    if key in seen:
                        raise ValueError(f"Duplicate evaluation recording {key}.")
                    seen.add(key)
                    if (str(item["speaker_id"]) != str(row.speaker_id) or not audio
                            or not audio.startswith(b"fLaC") or not tokens(item["text"] or "")):
                        records.append({"clip_id": row.clip_id, "status": "invalid_source_record"})
                        continue
                    path = folder / row.filename
                    path.write_bytes(audio)
                    references[row.clip_id] = item["text"]
                    records.append({"clip_id": row.clip_id, "status": "ok", "audio_path": str(path),
                                    "audio_bytes": len(audio), "audio_sha256": hashlib.sha256(audio).hexdigest()})
        except OSError as error:
            for key in selected_ids - seen:
                records.append({"clip_id": subset.loc[key, "clip_id"], "status": "shard_failed", "error": str(error)})
            seen = selected_ids
        finally:
            partial.unlink(missing_ok=True)
        for key in selected_ids - seen:
            records.append({"clip_id": subset.loc[key, "clip_id"], "status": "missing_from_shard"})
    retrieval = pd.DataFrame(records).set_index("clip_id").sort_index()
    successful = retrieval.status.eq("ok")
    if successful.any():
        duplicate = retrieval.loc[successful, "audio_sha256"].duplicated(keep=False)
        retrieval.loc[duplicate[duplicate].index, "status"] = "duplicate_audio_bytes"
    retrieval.to_csv(output / "retrieval.csv")
    return retrieval, pd.Series(references, name="reference", dtype=str)


def _rank(features, tree_scores):
    frame = features.assign(risk_score=tree_scores)
    signals = risk_signals(frame)
    random_scores = pd.Series(np.random.default_rng(RANDOM_SEED).permutation(len(frame)), index=frame.index)
    return pd.DataFrame({"random": random_scores, "native_confidence": signals.whisper_confidence,
                         "boosted_trees": signals.boosted_trees, "hybrid_50_50": blend(signals, .50)})


def _metrics(scores, labels):
    reviewed = pd.DataFrame({method: scores.index.isin(review_queue(scores[method])) for method in METHODS},
                            index=scores.index)
    catches = reviewed.loc[labels].sum()
    table = pd.DataFrame({"evaluated": len(labels), "high_error_total": int(labels.sum()),
                          "reviewed": reviewed.sum(), "caught": catches,
                          "precision": catches / reviewed.sum(), "recall": catches / (labels.sum() or np.nan),
                          "missed": labels.sum() - catches,
                          "extra_caught_vs_native": catches - catches["native_confidence"]})
    return reviewed, table


def _bootstrap(scores, labels, speakers, draws=2000):
    """Sample whole readers with replacement, re-rank each draw, and reapply ceil(20%)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = list(speakers.groupby(speakers, sort=True).indices.values())
    if len(groups) < 2 or labels.sum() == 0:
        return pd.DataFrame(columns=["method", "unit", "lower_95", "upper_95", "draws"])
    outcomes = []
    group_count = len(groups)
    native = scores.native_confidence.to_numpy()
    trees = scores.boosted_trees.to_numpy()
    uniform = scores.random.to_numpy()
    target = labels.to_numpy(dtype=bool)
    for _ in range(draws):
        picks = np.concatenate([groups[i] for i in rng.integers(0, group_count, group_count)])
        # The original rows are clip-ID sorted. Stable sorting ties by original ID,
        # then by occurrence for repeat copies of the same reader.
        picks.sort(kind="stable")
        y = target[picks]
        budget = math.ceil(len(y) * .20)
        signals = pd.DataFrame({"boosted_trees": trees[picks], "whisper_confidence": native[picks]})
        methods = (uniform[picks], native[picks], trees[picks], blend(signals, .50).to_numpy())
        counts = np.array([y[np.argsort(-values, kind="stable")[:budget]].sum() for values in methods])
        delta = counts - counts[1]
        outcomes.append(np.stack([delta, delta / budget, delta / (y.sum() or np.nan)]))
    data = np.stack(outcomes)
    records = []
    for unit_index, unit in enumerate(("detections", "precision", "recall")):
        for method_index, method in enumerate(METHODS):
            if method == "native_confidence":
                continue
            valid = data[:, unit_index, method_index]
            valid = valid[np.isfinite(valid)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
            records.append({"method": method, "unit": unit, "lower_95": low, "upper_95": high, "draws": len(valid)})
    return pd.DataFrame(records)


def _distribution(archived, live):
    confidence = FEATURE_NAMES[:5]
    records = []
    for name in confidence:
        for population, frame in (("Whispered English eligible development (archived)", archived),
                                  ("LibriSpeech sampled evaluation (live)", live)):
            values = frame[name].dropna()
            records.append({"population": population, "feature": name, "clips": len(frame),
                            "missing": len(frame) - len(values), "p10": values.quantile(.10),
                            "median": values.median(), "p90": values.quantile(.90)})
    return pd.DataFrame(records)


def _archive_features():
    from .train import ARCHIVE_PIPELINE, DATA_DIR, TEACHER, get_shard

    columns = ["clip_id", "segments", "whisper_text", "duration_s", "teacher_model", "pipeline_version",
               "compute_type", "runner", "language_bcp47"]
    frame = pq.read_table(get_shard("dev", 0, DATA_DIR), columns=columns).to_pandas().set_index("clip_id")
    frame = frame.loc[(frame.teacher_model.eq(TEACHER) & frame.runner.eq("faster-whisper")
                       & frame.compute_type.eq("float16") & frame.pipeline_version.eq(ARCHIVE_PIPELINE)
                       & frame.language_bcp47.eq("en") & frame.duration_s.between(0, 30, inclusive="right"))]
    return frame.rename(columns={"whisper_text": "transcript"})


def _run_clips(retrieval, output):
    input_path = output / "live_input.json"
    live_path = output / "live_whisper.json"
    retrieval.loc[retrieval.status.eq("ok"), ["audio_path"]].reset_index().to_json(input_path, orient="records")
    env = dict(os.environ, WHISPERWATCH_ASR_DEVICE="cuda", WHISPERWATCH_ASR_COMPUTE_TYPE="float16")
    library_dirs = [str(Path(next(iter(find_spec(name).submodule_search_locations))) / "lib")
                    for name in ("nvidia.cublas", "nvidia.cudnn")]
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [*library_dirs, env.get("LD_LIBRARY_PATH")]))
    subprocess.run([sys.executable, "-m", "src.whisper_compatibility", str(input_path), str(live_path),
                    "--continue-on-error"], cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    payload = json.loads(live_path.read_text(encoding="utf-8"))
    return payload["settings"], pd.DataFrame(payload["rows"]).set_index("clip_id", verify_integrity=True).sort_index()


def run_approved(plan, boosted_zip, output):
    """Run only in hosted Colab, using the approved manifest and frozen executable inputs."""
    if not Path("/content").is_dir() or "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Use a hosted Google Colab GPU runtime.")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    for library, expected in EXPECTED_LIBRARIES.items():
        if version(library) != expected:
            raise ValueError(f"Install {library}=={expected} before decoding.")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 6_000_000_000:
        raise OSError("At least 6 GB of free runtime disk is needed after installing dependencies.")
    audit = json.loads((plan / "preflight.json").read_text(encoding="utf-8"))
    frozen_bytes = (plan / "manifest.csv").read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != audit["manifest_sha256"]:
        raise ValueError("The preflight manifest was changed.")
    if (audit["source_revision"] != SOURCE_REVISION or audit["whispered_revision"] != REVISION
            or any(audit["id_matches"].values()) or audit["whispered_sources"] != ["common_voice_17"]):
        raise ValueError("The approved source provenance or overlap checks did not pass.")
    manifest = pd.read_csv(io.BytesIO(frozen_bytes)).sort_values("clip_id")
    if len(manifest) != 500 or not manifest.clip_id.is_unique:
        raise ValueError("The frozen manifest must contain 500 unique recordings.")
    with ZipFile(boosted_zip) as bundle:
        model_bytes = bundle.read(MODEL_MEMBER)
    if hashlib.sha256(model_bytes).hexdigest() != MODEL_SHA256:
        raise ValueError("The supplied boosted model differs from the frozen artifact.")
    model_path = output / "model.joblib"
    model_path.write_bytes(model_bytes)
    artifact = load_model(model_path)
    model_path.unlink()
    if artifact["sklearn_version"] != version("scikit-learn"):
        raise ValueError("The frozen artifact requires the matching scikit-learn version.")

    retrieval, references = _retrieve(manifest, output)
    if not retrieval.status.eq("ok").any():
        raise RuntimeError("No usable audio was recovered. Inspect retrieval.csv.")
    settings, live = _run_clips(retrieval, output)
    status = retrieval.status.rename("retrieval_status").to_frame()
    errors = live.get("error", pd.Series(index=live.index, dtype=str))
    successful = live.index[errors.isna()]
    status["inference_error"] = errors
    status["inference_status"] = "not_run"
    status.loc[live.index, "inference_status"] = "failed"
    status.loc[successful, "inference_status"] = "ok"
    status.to_csv(output / "clip_status.csv")
    if not len(successful):
        raise RuntimeError("No usable live transcription outputs. Inspect clip_status.csv.")
    live = live.loc[successful]
    unsupported = live.index[~live.duration_s.between(0, 30, inclusive="right")]
    status.loc[unsupported, "inference_status"] = "unsupported_duration"
    status.to_csv(output / "clip_status.csv")
    live = live.drop(unsupported)
    if live.empty:
        raise RuntimeError("All decoded clips were outside the supported duration range.")
    features = _features(live)
    scores = _rank(features, score_features(artifact, features.to_dict("records")))

    # References enter evaluation only, after all four ranking signals exist.
    references = references.loc[live.index]
    wers = pd.Series([diagnostic_wer(ref, text) for ref, text in zip(references, live.transcript)], index=live.index)
    labels = wers.gt(.20)
    reviewed, metrics = _metrics(scores, labels)
    identity = manifest.set_index("clip_id").loc[live.index, ["speaker_id", "chapter_id", "group"]]
    comparisons = reviewed.add_prefix("review_")
    predictions = identity.join(features).join(scores.add_prefix("score_")).join(comparisons)
    predictions = predictions.assign(transcript=live.transcript, reference=references, wer=wers,
                                     true_high_error=labels, seconds=live.seconds)
    predictions.to_csv(output / "predictions.csv")
    metrics.to_csv(output / "budget_comparison.csv", index_label="method")
    intervals = _bootstrap(scores, labels, identity.speaker_id)
    intervals.to_csv(output / "paired_bootstrap.csv", index=False)
    distribution = _distribution(_features(_archive_features()), features)
    distribution.to_csv(output / "confidence_distribution.csv", index=False)
    for method, other in (("hybrid_50_50", "native_confidence"), ("native_confidence", "hybrid_50_50")):
        only = labels & reviewed[method] & ~reviewed[other]
        predictions.loc[only].sort_values("wer", ascending=False).to_csv(output / f"errors_only_{method}.csv")
    (output / "manifest.csv").write_bytes(frozen_bytes)
    (output / "preflight.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    run_info = {"sampled": len(manifest), "evaluated": len(live), "evaluated_speakers": int(identity.speaker_id.nunique()),
                "retrieval_status": status.retrieval_status.value_counts().to_dict(),
                "inference_status": status.inference_status.value_counts().to_dict(),
                "excluded_fraction": 1 - len(live) / len(manifest),
                "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
                "fixed_budget_rounding": "ceil(0.20 * successfully evaluated clips), ties ascending clip ID",
                "model_sha256": MODEL_SHA256, "median_inference_seconds": float(live.seconds.median()),
                "evaluated_inference_seconds": float(live.seconds.sum()),
                "settings": settings, "random_baseline_seed": RANDOM_SEED, "bootstrap_seed": BOOTSTRAP_SEED}
    (output / "runtime.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    catches = metrics.caught
    recall_gain = metrics.at["hybrid_50_50", "recall"] - metrics.at["native_confidence", "recall"]
    ci = intervals.loc[intervals.method.eq("hybrid_50_50") & intervals.unit.eq("recall")]
    interval_text = "unavailable" if ci.empty else f"{ci.lower_95.iloc[0]:+.1%} to {ci.upper_95.iloc[0]:+.1%}"
    report = ["# WhisperWatch live validation", "",
              f"Evaluated {len(live)}/{len(manifest)} clips from {run_info['evaluated_speakers']} readers. "
              f"Each method used {int(metrics.reviewed.iloc[0])} review slots. "
              f"The hybrid caught {catches['hybrid_50_50']} high-error clips, and native confidence caught "
              f"{catches['native_confidence']} of {int(labels.sum())}.", "",
              f"Hybrid recall difference: {recall_gain:+.1%} (percentage points expressed as a rate). "
              f"Paired reader-bootstrap 95% interval: {interval_text}.", "",
              "The archive-versus-live feature summary compares Common Voice with LibriSpeech. "
              "Differences there may reflect both the recordings and the decoder. See the earlier paired 80-clip "
              "diagnostic for a same-audio comparison.", "",
              "The comparison applies to this audiobook sample. Review the paired interval and failed-clip counts "
              "before deciding whether the hybrid adds enough value to keep."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics, intervals, run_info
