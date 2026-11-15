"""Bounded, development-only feasibility checks for a second speech recognizer."""

import hashlib
import json
import math
import re
import shutil
import tarfile
import unicodedata
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .ensemble_experiment import LABEL, blend, read_development, review_queue, risk_signals
from .train import DATA_DIR, REVISION, get_shard


AUDIO_REVISION = "8262c16bf297c87a9cd88c51997c4758ed7a8ba2"
AUDIO_REPO = "fsicoli/common_voice_17_0"
AUDIO_ARCHIVE = "audio/en/dev/en_dev_0.tar"
AUDIO_URL = f"https://huggingface.co/datasets/{AUDIO_REPO}/resolve/{AUDIO_REVISION}/{AUDIO_ARCHIVE}"
MAX_ARCHIVE_BYTES = 750_000_000
WAV2VEC_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
WHISPER_REPO = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
SEED = 20261114


def select_clips(boosted_zip, split_zip) -> pd.DataFrame:
    """Freeze 80 clips with distinct speakers using the prior split and hybrid queues."""
    dev = read_development(boosted_zip)
    with ZipFile(split_zip) as archive, archive.open("development_split.csv") as csv:
        partition = pd.read_csv(csv).set_index("clip_id")["partition"]
    if not partition.index.is_unique or set(partition.index) != set(dev.index):
        raise ValueError("The development predictions and partition must cover the same unique clip IDs.")
    if set(partition) != {"tuning", "verification"}:
        raise ValueError("Expected tuning and verification partitions.")
    columns = ["clip_id", "speaker_id", "duration_s", "sampling_rate_src", "audio_ref",
               "audio_sha256", "source_revision", "human_text", "whisper_text", "wer_whisper_vs_human",
               "teacher_model", "runner", "compute_type", "pipeline_version", "segments"]
    meta = pq.read_table(get_shard("dev", 0, DATA_DIR), columns=columns).to_pandas()
    meta = meta.set_index("clip_id", verify_integrity=True).loc[dev.index]
    data = dev.join(meta).assign(partition=partition)
    if data.speaker_id.isna().any() or data.speaker_id.eq("").any():
        raise ValueError("Missing speaker ID in pilot source.")
    groups = data.groupby("partition").speaker_id.apply(set)
    if groups["tuning"] & groups["verification"]:
        raise ValueError("Speakers overlap across partitions.")
    if not data[LABEL].eq(data.wer_whisper_vs_human.gt(0.20).astype(int)).all():
        raise ValueError("The exported labels differ from the archived WER.")
    counts = {"tuning": (8_207, 1_642, 676), "verification": (8_152, 1_631, 715)}
    data["hybrid_flagged"] = False
    for name, subset in data.groupby("partition"):
        signals = risk_signals(subset)
        data.loc[subset.index, "hybrid_rank"] = blend(signals, 0.50)
        data.loc[subset.index, "confidence_rank"] = signals.whisper_confidence.rank(pct=True)
        queue = review_queue(data.loc[subset.index, "hybrid_rank"])
        if (len(subset), len(queue), int(subset.loc[queue, LABEL].sum())) != counts[name]:
            raise ValueError("The saved split and scores do not reproduce the existing 50/50 hybrid run.")
        data.loc[queue, "hybrid_flagged"] = True
    data["duration_band"] = np.where(data.duration_s <= data.duration_s.median(), "shorter", "longer")
    strata = ["partition", LABEL, "hybrid_flagged", "duration_band"]
    picks = []
    used = set()
    for _, group in data.groupby(strata, sort=True):
        candidates = group.sample(frac=1, random_state=SEED).drop_duplicates("speaker_id")
        chosen = candidates.loc[~candidates.speaker_id.isin(used)].head(5)
        if len(chosen) < 5:
            raise ValueError("A pilot stratum lacks five unused speakers.")
        picks.append(chosen)
        used.update(chosen.speaker_id)
    selected = pd.concat(picks).sort_index()
    if len(selected) != 80:
        raise ValueError("Expected 16 pilot strata with five clips each.")
    selected["live_whisper_sample"] = False
    for _, group in selected.groupby(strata[:-1]):
        selected.loc[group.sample(n=1, random_state=SEED).index, "live_whisper_sample"] = True
    selected["archive_segment_count"] = selected.segments.map(len)
    return selected.drop(columns="segments").reset_index()


def _audio_name(row) -> str:
    ref = row.audio_ref
    identity = (ref["dataset"], ref["config"], ref["split"], int(ref["shard"]), row.source_revision)
    if identity != (AUDIO_REPO, "en", "dev", 0, AUDIO_REVISION):
        raise ValueError(f"Unexpected audio archive for {row.clip_id}.")
    filename = ref["file"]
    if not re.fullmatch(r"common_voice_en_\d+\.mp3", filename) or row.clip_id.rsplit("/", 1)[1] != filename[:-4]:
        raise ValueError(f"Audio reference mismatch for {row.clip_id}.")
    return filename


class BoundedStream:
    """Count audio archive payload bytes and cap the transfer."""

    def __init__(self, source, limit):
        self.source = source
        self.limit = limit
        self.bytes_read = 0

    def read(self, length):
        remaining = self.limit - self.bytes_read
        if remaining <= 0:
            raise OSError("Audio archive transfer reached its configured limit.")
        data = self.source.read(min(length, remaining))
        self.bytes_read += len(data)
        return data


def retrieve_audio(selected: pd.DataFrame, folder: Path):
    """Adapt the publisher's forward tar scan with a pinned source and byte cap."""
    import av

    folder.mkdir(parents=True, exist_ok=True)
    wanted = {_audio_name(row): row for row in selected.itertuples(index=False)}
    if len(wanted) != len(selected):
        raise ValueError("The pilot contains duplicate source filenames.")
    records = {row.clip_id: {"clip_id": row.clip_id, "retrieval_status": "missing",
                            "retrieval_error": "Archive member not found"}
               for row in wanted.values()}
    start = perf_counter()
    stream = None
    size = None
    error = ""
    try:
        with urlopen(Request(AUDIO_URL, method="HEAD"), timeout=30) as response:
            size = int(response.headers.get("Content-Length", 0))
        print(f"Pinned audio archive size {size:,} bytes. Transfer cap {MAX_ARCHIVE_BYTES:,} bytes.", flush=True)
        if not 0 < size <= MAX_ARCHIVE_BYTES:
            raise OSError("Archive size is unknown or exceeds the pilot transfer cap.")
        remaining = set(wanted)
        with urlopen(AUDIO_URL, timeout=60) as response:
            stream = BoundedStream(response, MAX_ARCHIVE_BYTES)
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or member.name != "en_dev_0/" + Path(member.name).name:
                        continue
                    filename = Path(member.name).name
                    if filename not in remaining:
                        continue
                    row = wanted[filename]
                    record = records[row.clip_id]
                    remaining.remove(filename)
                    try:
                        if not 0 < member.size <= 2_000_000:
                            raise ValueError("Clip size is outside pilot limits.")
                        with archive.extractfile(member) as source:
                            audio = source.read()
                        if len(audio) != member.size:
                            raise ValueError("Truncated clip member.")
                        digest = hashlib.sha256(audio).hexdigest()
                        has_source_hash = isinstance(row.audio_sha256, str) and bool(row.audio_sha256)
                        if has_source_hash and digest != row.audio_sha256:
                            raise ValueError("Clip checksum differs from archive metadata.")
                        with av.open(BytesIO(audio)) as container:
                            codec = container.streams.audio[0].codec_context
                            sample_rate = codec.sample_rate
                            format_name = container.format.name
                            samples = sum(frame.samples for frame in container.decode(audio=0))
                        if not sample_rate or not samples:
                            raise ValueError("The audio member has no decodable samples.")
                        duration = samples / sample_rate
                        delta = duration - row.duration_s
                        record.update(audio_duration_s=duration, duration_delta_s=delta, sample_rate=sample_rate,
                                      audio_format=format_name, clip_bytes=len(audio), sha256=digest,
                                      archive_hash_available=has_source_hash)
                        if format_name != "mp3" or abs(delta) > 0.15:
                            raise ValueError("MP3 format or decoded duration differs from the expected clip.")
                        if pd.notna(row.sampling_rate_src) and sample_rate != row.sampling_rate_src:
                            raise ValueError("Decoded sample rate differs from archive metadata.")
                        path = folder / filename
                        path.write_bytes(audio)
                        record.update(retrieval_status="ok", retrieval_error="", audio_path=str(path))
                    except (ValueError, IndexError, OSError, av.error.FFmpegError) as exc:
                        record.update(retrieval_status="invalid", retrieval_error=str(exc))
                    if not remaining:
                        break
    except (OSError, ValueError, tarfile.TarError) as exc:
        error = str(exc)
        for record in records.values():
            if record["retrieval_status"] == "missing":
                record.update(retrieval_status="archive_error", retrieval_error=error)
    retrieved = pd.DataFrame(records.values())
    return retrieved, {
        "archive_url": AUDIO_URL, "archive_revision": AUDIO_REVISION,
        "archive_bytes": size, "transferred_payload_bytes": stream.bytes_read if stream else 0,
        "retrieval_wall_s": perf_counter() - start, "retrieval_error": error,
        "retrieval_counts": retrieved.retrieval_status.value_counts().to_dict(),
    }


def tokens(text: str) -> list[str]:
    """Normalize case, Unicode, punctuation and whitespace for English text."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text).split()


def edit_distance(left: list[str], right: list[str]) -> int:
    """Compute word Levenshtein distance using one row of memory."""
    previous = list(range(len(right) + 1))
    for row, source in enumerate(left, start=1):
        current = [row]
        for col, target in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (source != target)))
        previous = current
    return previous[-1]


def disagreement(first: str, second: str) -> float:
    """Symmetric normalized edit distance. Two empty transcripts have zero distance."""
    left, right = tokens(first), tokens(second)
    return edit_distance(left, right) / max(len(left), len(right), 1)


def diagnostic_wer(reference: str, transcript: str) -> float:
    reference_words = tokens(reference)
    if not reference_words:
        return math.nan
    return edit_distance(reference_words, tokens(transcript)) / len(reference_words)


def run_wav2vec(selected: pd.DataFrame):
    """Run the pinned ASR-ready model one clip at a time and time the second pass."""
    import torch
    from faster_whisper.audio import decode_audio
    from transformers import AutoModelForCTC, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Select a hosted Colab GPU runtime for model inference.")
    device = torch.device("cuda")
    started = perf_counter()
    processor = AutoProcessor.from_pretrained(WAV2VEC_MODEL, revision=WAV2VEC_REVISION)
    model = AutoModelForCTC.from_pretrained(
        WAV2VEC_MODEL, revision=WAV2VEC_REVISION, use_safetensors=True
    ).to(device).eval()
    torch.cuda.synchronize()
    setup_s = perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for row in selected.itertuples(index=False):
        record = {
            "clip_id": row.clip_id, "inference_status": "skipped_retrieval", "inference_error": "",
            "second_asr_text": None, "disagreement": math.nan,
            "second_asr_wer": math.nan, "second_asr_seconds": math.nan,
        }
        if row.retrieval_status == "ok":
            start = perf_counter()
            try:
                waveform = decode_audio(row.audio_path, sampling_rate=16_000)
                inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt").to(device)
                with torch.inference_mode():
                    text = processor.batch_decode(model(**inputs).logits.argmax(dim=-1))[0].strip()
                torch.cuda.synchronize()
                record.update(inference_status="ok", second_asr_text=text,
                              disagreement=disagreement(row.whisper_text, text),
                              second_asr_wer=diagnostic_wer(row.human_text, text))
            except (RuntimeError, ValueError, OSError) as exc:
                record.update(inference_status="failed", inference_error=str(exc))
            record["second_asr_seconds"] = perf_counter() - start
        rows.append(record)
    usage = {
        "model": WAV2VEC_MODEL, "revision": WAV2VEC_REVISION, "device": torch.cuda.get_device_name(),
        "model_setup_s": setup_s, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "disk_free_bytes": shutil.disk_usage("/content").free,
    }
    return pd.DataFrame(rows), usage


def run_live_check(sample_path: Path, output_path: Path):
    """Run the app's decoder with the pinned Large V3 weights in a clean GPU process."""
    from unittest.mock import patch

    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    from . import asr
    from .features import extract_features

    frame = pd.read_json(sample_path, orient="records")
    started = perf_counter()
    local_model = snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION,
                                    allow_patterns=["*.json", "model.bin"])
    model = WhisperModel(local_model, device="cuda", compute_type="float16")
    setup_s = perf_counter() - started
    rows = []
    with patch.object(asr, "_whisper_model", return_value=model):
        for row in frame.itertuples(index=False):
            record = {"clip_id": row.clip_id, "live_status": "ok"}
            started = perf_counter()
            try:
                result = asr.transcribe_audio(row.audio_path)
                features = extract_features(result["segments"], result["transcript"], result["duration_s"])
                record.update(live_whisper_text=result["transcript"],
                              archive_live_disagreement=disagreement(row.whisper_text, result["transcript"]),
                              live_disagreement=disagreement(result["transcript"], row.second_asr_text),
                              live_segment_count=features["num_segments"],
                              live_duration_s=result["duration_s"],
                              live_avg_logprob=features["avg_logprob"]
                              if math.isfinite(features["avg_logprob"]) else None)
            except (ValueError, OSError, RuntimeError) as exc:
                record.update(live_status="failed", live_error=str(exc))
            record["live_seconds"] = perf_counter() - started
            rows.append(record)
    output_path.write_text(json.dumps({
        "model": WHISPER_REPO, "revision": WHISPER_REVISION, "setup_s": setup_s,
        "settings": {"language": "en", "task": "transcribe", "compute_type": "float16",
                     "vad_filter": False, "beam_size": 5, "word_timestamps": False},
        "rows": rows,
    }, allow_nan=False), encoding="utf-8")


def diagnostics(scored: pd.DataFrame):
    """Summarize complementarity within the preselected feasibility strata."""
    processed = scored.loc[scored.inference_status.eq("ok")].copy()
    groups = ["partition", "hybrid_flagged", LABEL]
    cohort = processed.groupby(groups).agg(
        clips=("clip_id", "size"), median_disagreement=("disagreement", "median"),
        median_second_wer=("second_asr_wer", "median"),
    )
    # Compare positives and negatives with similar existing hybrid and confidence ranks.
    # Use rank bins fixed in advance, with both partitions kept distinct.
    processed["risk_bin"] = np.minimum((processed.hybrid_rank * 5).astype(int), 4)
    processed["confidence_bin"] = np.minimum((processed.confidence_rank * 5).astype(int), 4)
    matched = []
    for cells, group in processed.groupby(["partition", "hybrid_flagged", "risk_bin", "confidence_bin"]):
        high = group.loc[group[LABEL].eq(1), "disagreement"].to_numpy()
        low = group.loc[group[LABEL].eq(0), "disagreement"].to_numpy()
        if len(high) and len(low):
            difference = high[:, None] - low
            matched.append(dict(zip(["partition", "hybrid_flagged", "risk_bin", "confidence_bin"], cells),
                                pairs=difference.size, high_exceeds_low=int((difference > 0).sum()),
                                tied=int((difference == 0).sum())))
    examples = []
    for kind, mask, ascending in [
        ("missed high error with disagreement", processed[LABEL].eq(1) & ~processed.hybrid_flagged, False),
        ("low error with disagreement", processed[LABEL].eq(0), False),
        ("similar transcripts, both high error", processed[LABEL].eq(1) & processed.second_asr_wer.gt(.20), True),
    ]:
        examples.append(processed.loc[mask].sort_values(["disagreement", "clip_id"], ascending=[ascending, True])
                        .head(3).assign(example_type=kind))
    return cohort, pd.DataFrame(matched), pd.concat(examples)
