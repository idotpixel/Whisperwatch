"""Compare archived and live review queues for the fixed Colab pilot."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import asr
from .ensemble_experiment import blend, read_development, review_queue, risk_signals
from .features import FEATURE_NAMES, extract_features
from .inference import load_model, score_features
from .second_asr_pilot import WHISPER_REPO, WHISPER_REVISION, diagnostic_wer, disagreement
from .train import ARCHIVE_PIPELINE, DATA_DIR, REVISION, TEACHER, get_shard


SEGMENT_FIELDS = ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio", "temperature")
MODEL_MEMBER = "models/whisperwatch_boosted.joblib"


def _read_pilot(project):
    """Fail before inference unless all 80 files match the original retrieval hashes."""
    manifest = pd.read_csv(project / "selected_clips.csv", usecols=["clip_id", "speaker_id", "partition"])
    manifest = manifest.set_index("clip_id", verify_integrity=True).sort_index()
    if (len(manifest) != 80 or manifest.speaker_id.nunique() != 80
            or not manifest.index.str.startswith("common_voice_17/en/dev/").all()):
        raise ValueError("Expected 80 distinct development clips and speakers from the pilot.")
    retrieval = pd.read_csv(project / "retrieval.csv").set_index("clip_id", verify_integrity=True)
    recognition = pd.read_csv(project / "second_asr_results.csv", keep_default_na=False).set_index(
        "clip_id", verify_integrity=True
    )
    if set(manifest.index) != set(retrieval.index) or set(manifest.index) != set(recognition.index):
        raise ValueError("The pilot manifest, retrieval, and transcription exports differ.")
    retrieval = retrieval.loc[manifest.index]
    recognition = recognition.loc[manifest.index]
    if not retrieval.retrieval_status.eq("ok").all() or not recognition.inference_status.eq("ok").all():
        raise ValueError("All 80 archived Wav2Vec2 outputs and audio files are required.")
    paths = []
    for clip_id, row in retrieval.iterrows():
        name = clip_id.rsplit("/", 1)[-1] + ".mp3"
        path = project / "audio" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Missing or changed original pilot audio: {name}. Restore the pilot audio folder.")
        paths.append(str(path))
    return manifest.assign(audio_path=paths, second_asr_text=recognition.second_asr_text)


def _run_live(input_path, output_path, *, continue_on_error=False):
    """Invoke the production transcriber with the pilot's exact pinned GPU weights."""
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    started = perf_counter()
    local = snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION, allow_patterns=["*.json", "model.bin"])
    model = WhisperModel(local, device="cuda", compute_type="float16")
    settings = {"model": WHISPER_REPO, "revision": WHISPER_REVISION, "compute_type": "float16",
                "setup_seconds": perf_counter() - started,
                "libraries": {name: version(name) for name in ("faster-whisper", "ctranslate2", "av")}}
    rows = []
    transcribe = model.transcribe

    def capture_options(*args, **kwargs):
        segments, info = transcribe(*args, **kwargs)
        settings.update(language=info.language, task=kwargs.get("task"),
                        decoding=asdict(info.transcription_options),
                        vad_filter=info.vad_options is not None,
                        audio_loading="faster-whisper opens the original MP3 with PyAV and decodes mono 16 kHz")
        return segments, info

    frame = pd.read_json(input_path, orient="records")
    with patch.object(asr, "_whisper_model", return_value=model), patch.object(model, "transcribe", capture_options):
        for row in frame.itertuples(index=False):
            started = perf_counter()
            try:
                audio = asr.transcribe_audio(row.audio_path)
            except Exception as error:
                if continue_on_error and isinstance(error, (ValueError, OSError, RuntimeError)):
                    rows.append({"clip_id": row.clip_id, "error": str(error), "seconds": perf_counter() - started})
                    continue
                output_path.write_text(json.dumps({"completed": rows, "failed_clip": row.clip_id,
                                                   "error": str(error)}, default=str), encoding="utf-8")
                raise RuntimeError(f"Live transcription failed for {row.clip_id}") from error
            rows.append({"clip_id": row.clip_id, "transcript": audio["transcript"],
                         "duration_s": audio["duration_s"], "seconds": perf_counter() - started,
                         "segments": [{name: getattr(segment, name) for name in SEGMENT_FIELDS}
                                      for segment in audio["segments"]]})
    output_path.write_text(json.dumps({"settings": settings, "rows": rows}, default=str), encoding="utf-8")


def _features(frame):
    return pd.DataFrame(
        [extract_features(row.segments, row.transcript, row.duration_s) for row in frame.itertuples()],
        index=frame.index, columns=FEATURE_NAMES,
    )


def _drift(archived, live):
    rows = []
    for name in archived:
        old, new = archived[name], live[name]
        delta = new - old
        iqr = old.quantile(.75) - old.quantile(.25)
        changed = ~np.isclose(old, new, atol=1e-8, rtol=1e-7, equal_nan=True)
        rows.append({"feature": name, "archived_median": old.median(), "live_median": new.median(),
                     "median_change": delta.median(), "median_absolute_change": delta.abs().median(),
                     "p90_absolute_change": delta.abs().quantile(.9), "changed_clips": int(changed.sum()),
                     "lower_live_clips": int(((delta < 0) & changed).sum()),
                     "higher_live_clips": int(((delta > 0) & changed).sum()),
                     "archive_missing": int(old.isna().sum()), "live_missing": int(new.isna().sum()),
                     "median_absolute_change_in_archived_iqrs": delta.abs().median() / iqr if iqr > 0 else np.nan,
                     "spearman": _spearman(old, new)})
    return pd.DataFrame(rows).set_index("feature")


def _spearman(left, right):
    """Pearson correlation of average ranks over the same nonmissing pairs."""
    paired = pd.concat([left, right], axis=1).dropna()
    if (paired.nunique() < 2).any():
        return np.nan
    ranks = paired.rank(method="average")
    return float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1]))


def _compare(archived, live, pilot, artifact, labels, expected, output):
    old, new = _features(archived), _features(live)
    comparison = pilot[["speaker_id", "partition", "second_asr_text"]].copy()
    for prefix, frame, features in (("archived", archived, old), ("live", live, new)):
        scores = pd.Series(score_features(artifact, features.to_dict("records")), index=features.index)
        if prefix == "archived":
            np.testing.assert_allclose(scores, expected.risk_score, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(features.avg_logprob, expected.avg_logprob, equal_nan=True)
        signals = risk_signals(pd.DataFrame({"risk_score": scores, "avg_logprob": features.avg_logprob}))
        hybrid = blend(signals, .50)
        comparison = comparison.join(features.add_prefix(prefix + "_"))
        comparison[prefix + "_text"] = frame.transcript
        comparison[prefix + "_tree_risk"] = scores
        comparison[prefix + "_hybrid"] = hybrid
        ordered = hybrid.sort_index().sort_values(ascending=False, kind="stable")
        comparison[prefix + "_rank"] = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
        comparison[prefix + "_review"] = comparison.index.isin(review_queue(hybrid))
        comparison[prefix + "_second_asr_disagreement"] = [
            disagreement(text, second) for text, second in zip(frame.transcript, pilot.second_asr_text)
        ]
    comparison["archived_vs_live_disagreement"] = [
        disagreement(old_text, new_text) for old_text, new_text in zip(archived.transcript, live.transcript)
    ]
    for prefix, frame in (("archived", archived), ("live", live)):
        comparison[prefix + "_first_segment_start_s"] = frame.segments.map(
            lambda segments: segments[0]["start"] if len(segments) else np.nan
        )
        comparison[prefix + "_last_segment_end_s"] = frame.segments.map(
            lambda segments: segments[-1]["end"] if len(segments) else np.nan
        )
    # Evaluation starts after the scores and review lists have been frozen above.
    comparison["true_high_error_archived"] = labels.true_high_error
    for prefix, frame in (("archived", archived), ("live", live)):
        comparison[prefix + "_wer_common_normalization"] = [
            diagnostic_wer(reference, text) for reference, text in zip(labels.human_text, frame.transcript)
        ]
    old_queue = comparison.archived_review
    new_queue = comparison.live_review
    high_error = comparison.true_high_error_archived.eq(1)
    delta = comparison.live_second_asr_disagreement - comparison.archived_second_asr_disagreement
    disagreement_rho = _spearman(comparison.archived_second_asr_disagreement, comparison.live_second_asr_disagreement)
    archived_target = comparison.archived_wer_common_normalization.gt(.20)
    live_target = comparison.live_wer_common_normalization.gt(.20)
    summary = {
        "clips": len(comparison), "reviewed_per_queue": int(old_queue.sum()),
        "hybrid_spearman": _spearman(comparison.archived_hybrid, comparison.live_hybrid),
        "boosted_spearman": _spearman(comparison.archived_tree_risk, comparison.live_tree_risk),
        "confidence_spearman": _spearman(-old.avg_logprob, -new.avg_logprob),
        "shared_top_clips": int((old_queue & new_queue).sum()),
        "overlap_fraction": float((old_queue & new_queue).sum() / old_queue.sum()),
        "archived_high_error_labels": int(high_error.sum()),
        "archived_queue_caught_fixed_labels": int((high_error & old_queue).sum()),
        "live_queue_caught_fixed_labels": int((high_error & new_queue).sum()),
        "high_error_shared": int((high_error & old_queue & new_queue).sum()),
        "high_error_archived_queue_only": int((high_error & old_queue & ~new_queue).sum()),
        "high_error_live_queue_only": int((high_error & new_queue & ~old_queue).sum()),
        "same_normalizer_archived_high_error_total": int(archived_target.sum()),
        "same_normalizer_archived_queue_catches_own_errors": int((archived_target & old_queue).sum()),
        "same_normalizer_live_high_error_total": int(live_target.sum()),
        "same_normalizer_live_queue_catches_own_errors": int((live_target & new_queue).sum()),
        "same_normalizer_archived_queue_catches_live_errors": int((live_target & old_queue).sum()),
        "same_normalizer_high_error_labels_that_changed": int(archived_target.ne(live_target).sum()),
        "normalized_transcripts_identical": int(comparison.archived_vs_live_disagreement.eq(0).sum()),
        "median_archived_vs_live_text_disagreement": float(comparison.archived_vs_live_disagreement.median()),
        "clips_with_different_segment_counts": int(old.num_segments.ne(new.num_segments).sum()),
        "median_absolute_first_segment_start_change_s": float((comparison.live_first_segment_start_s
                                                                - comparison.archived_first_segment_start_s).abs().median()),
        "median_absolute_last_segment_end_change_s": float((comparison.live_last_segment_end_s
                                                            - comparison.archived_last_segment_end_s).abs().median()),
        "second_asr_disagreement_median_change": float(delta.median()),
        "second_asr_disagreement_median_absolute_change": float(delta.abs().median()),
        "second_asr_disagreement_p90_absolute_change": float(delta.abs().quantile(.9)),
        "second_asr_disagreement_changes_above_0_10": int(delta.abs().gt(.10 + 1e-12).sum()),
        "second_asr_disagreement_spearman": disagreement_rho,
    }
    comparison.assign(second_asr_disagreement_change=delta).to_csv(output / "clip_comparison.csv")
    _drift(old, new).to_csv(output / "feature_drift.csv")
    return {name: None if pd.isna(value) else value for name, value in summary.items()}


def run(project, boosted_zip, output):
    """Run the compatibility check from the pilot's existing hosted runtime."""
    if not Path("/content").is_dir() or "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Run this follow-up in the original hosted Google Colab GPU runtime.")
    pilot = _read_pilot(project)
    subprocess.run(["nvidia-smi", "-L"], check=True)
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(boosted_zip) as archive:
        artifact_path = output / "frozen_model.joblib"
        raw_model = archive.read(MODEL_MEMBER)
        artifact_path.write_bytes(raw_model)
    try:
        artifact = load_model(artifact_path)
    finally:
        artifact_path.unlink()
    if (artifact["dataset_revision"] != REVISION or artifact["sklearn_version"] != version("scikit-learn")
            or type(artifact["pipeline"]).__name__ != "HistGradientBoostingClassifier"):
        raise ValueError("Expected the original boosted-tree model and its scikit-learn version.")
    expected = read_development(boosted_zip).loc[pilot.index]
    columns = ["clip_id", "split", "teacher_model", "runner", "compute_type", "pipeline_version",
               "whisper_text", "duration_s", "segments", "human_text", "wer_whisper_vs_human"]
    archive_data = pq.read_table(get_shard("dev", 0, DATA_DIR), columns=columns,
                                 filters=[("clip_id", "in", pilot.index.tolist())]).to_pandas()
    archive_data = archive_data.set_index("clip_id", verify_integrity=True).loc[pilot.index]
    for column, value in (("split", "dev"), ("teacher_model", TEACHER), ("runner", "faster-whisper"),
                          ("compute_type", "float16"), ("pipeline_version", ARCHIVE_PIPELINE)):
        if not archive_data[column].eq(value).all():
            raise ValueError(f"Unexpected archived {column}.")
    archived = archive_data[["whisper_text", "duration_s", "segments"]].rename(columns={"whisper_text": "transcript"})
    labels = archive_data[["human_text"]].assign(true_high_error=expected.true_high_error)
    if not labels.true_high_error.eq(archive_data.wer_whisper_vs_human.gt(.20)).all():
        raise ValueError("The original development labels and archived WER differ.")
    old_features = _features(archived)
    np.testing.assert_allclose(score_features(artifact, old_features.to_dict("records")),
                               expected.risk_score, rtol=1e-10, atol=1e-12)

    input_path = output / "live_input.json"
    pilot[["audio_path"]].reset_index().to_json(input_path, orient="records")
    live_path = output / "live_whisper.json"
    env = dict(os.environ, WHISPERWATCH_ASR_DEVICE="cuda", WHISPERWATCH_ASR_COMPUTE_TYPE="float16")
    lib_dirs = [str(Path(next(iter(find_spec(name).submodule_search_locations))) / "lib")
                for name in ("nvidia.cublas", "nvidia.cudnn")]
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [*lib_dirs, env.get("LD_LIBRARY_PATH")]))
    try:
        subprocess.run([sys.executable, "-m", "src.whisper_compatibility", str(input_path), str(live_path)],
                       cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    finally:
        input_path.unlink(missing_ok=True)

    raw = json.loads(live_path.read_text(encoding="utf-8"))
    live = pd.DataFrame(raw["rows"]).set_index("clip_id", verify_integrity=True).loc[pilot.index]
    archived_segments = {clip_id: [{name: segment.get(name) for name in SEGMENT_FIELDS} for segment in segments]
                         for clip_id, segments in archived.segments.items()}
    (output / "archived_segments.json").write_text(json.dumps(archived_segments, default=str), encoding="utf-8")
    summary = _compare(archived, live, pilot, artifact, labels, expected, output)
    summary.update(model_sha256=hashlib.sha256(raw_model).hexdigest(), archive_dataset_revision=REVISION,
                   archive_pipeline=ARCHIVE_PIPELINE, live_settings=raw["settings"],
                   median_live_seconds=float(live.seconds.median()), total_live_seconds=float(live.seconds.sum()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_report(summary, pd.read_csv(output / "feature_drift.csv"), output / "report.md")
    return summary


def _write_report(s, drift, path):
    def rho(name):
        value = s[name]
        return f"{value:.3f}" if value is not None else "undefined (constant or missing scores)"

    count = s["reviewed_per_queue"]
    swapped = count - s["shared_top_clips"]
    lines = [
        "# Archived and live Whisper ranking compatibility", "",
        f"The two top-{count} review lists shared **{s['shared_top_clips']}/{count}** clips "
        f"({s['overlap_fraction']:.1%}). Overall hybrid Spearman correlation was **{rho('hybrid_spearman')}**. "
        f"Boosted-tree and native-confidence rank correlations were {rho('boosted_spearman')} and "
        f"{rho('confidence_spearman')}, respectively. Switching to live outputs replaced "
        f"{swapped} of the {count} review slots.", "",
        f"Against the same {s['archived_high_error_labels']} original high-error labels, the archived queue "
        f"caught **{s['archived_queue_caught_fixed_labels']}** and the live queue caught "
        f"**{s['live_queue_caught_fixed_labels']}**. The archived and live queues uniquely caught "
        f"{s['high_error_archived_queue_only']} and {s['high_error_live_queue_only']}, respectively. "
        f"With both queues judged against fresh-transcript errors using a common text normalizer, "
        f"they caught {s['same_normalizer_archived_queue_catches_live_errors']} and "
        f"{s['same_normalizer_live_queue_catches_own_errors']} of {s['same_normalizer_live_high_error_total']}. "
        f"Using that normalizer on both transcripts, {s['same_normalizer_high_error_labels_that_changed']} "
        "clips crossed the 20% WER boundary between outputs.", "",
        f"{s['normalized_transcripts_identical']}/{s['clips']} transcripts matched after normalization. "
        f"The segment count changed on {s['clips_with_different_segment_counts']} clips. "
        f"Second-ASR disagreement rank correlation was {rho('second_asr_disagreement_spearman')}. "
        f"Its median and 90th-percentile absolute changes were {s['second_asr_disagreement_median_absolute_change']:.3f} "
        f"and {s['second_asr_disagreement_p90_absolute_change']:.3f}. "
        f"{s['second_asr_disagreement_changes_above_0_10']} clips shifted by more than 0.10.", "",
        "Largest typical feature changes relative to the archived feature's middle 50% range:", "",
    ]
    key = "median_absolute_change_in_archived_iqrs"
    for row in drift.dropna(subset=[key]).nlargest(3, key).to_dict("records"):
        lines.append(f"- `{row['feature']}` shifted by a median of {row['median_absolute_change']:.4g} "
                     f"in its own units ({row[key]:.2f} archived interquartile ranges). "
                     f"It was lower on {row['lower_live_clips']} and higher on {row['higher_live_clips']} clips.")
    lines.extend([
        "", "The archive records Large V3, faster-whisper, float16, and its pipeline version. "
        "The live run records the effective decoder options and input path in `summary.json`. "
        "Incomplete archived settings prevent attributing drift to a specific checkpoint conversion, "
        "decoding option, VAD setup, or preprocessing step.", "",
        "These 80 clips were selected for feasibility. The counts describe this batch only. "
        "Compatibility depends on both queue turnover and which errors moved in or out. The second-ASR "
        "run should stay at pilot scale pending review of those changes and the disagreement drift.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe the fixed pilot with live Whisper GPU.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    _run_live(args.input, args.output, continue_on_error=args.continue_on_error)
