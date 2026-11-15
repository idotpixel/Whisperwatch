# WhisperWatch evaluation handoff

Send Member C `whisperwatch-evaluation-handoff.zip` from the repository’s ignored `tmp/` folder. It contains one normalized CSV per evaluation, `verified_counts.csv`, and `provenance.json` with SHA-256 hashes of the inputs. The original Colab archives remain in Downloads. Keep the prediction data within the team. It includes source reference text and recording identifiers, with no raw audio. Attribute the corpora when reusing examples. [Whispered and Common Voice](https://huggingface.co/datasets/burakaydinofficial/Whispered) · [LibriSpeech, CC BY 4.0](https://www.openslr.org/12/) · [AMI IHM, CC BY 4.0](https://huggingface.co/datasets/edinburghcstr/ami)

| Export | Original inputs | Dataset role | Clips | High error | Reviewed | Native caught | Original hybrid caught |
|---|---|---|---:|---:|---:|---:|---:|
| `common_voice_dev_verification.csv` | Boosted results ZIP, ensemble speaker-split ZIP and pinned English dev metadata | Explored, speaker-separated archived development verification | 8,152 | 1,075 | 1,631 | 705 | 715 |
| `librispeech_live.csv` | Live validation ZIP | External live audiobook holdout | 499 | 22 | 100 | 14 | 15 |
| `ami_retraining_holdout.csv` | Live retraining ZIP | Fresh live AMI holdout for the earlier experiment. Four linked meeting groups | 750 | 256 | 150 | 89 | 91 |
| `ami_audio_aware_holdout.csv` | Audio-aware results ZIP | Separate fresh live AMI holdout. Sixteen linked meeting groups | 640 | 244 | 128 | 81 | 80 |

Additional verified rows include 4 random detections on LibriSpeech, 52 random detections on the first AMI holdout, and 83 detections for the audio-aware hybrid on the latest AMI holdout. On the latest AMI holdout the original tree score was recomputed from the exact frozen artifact and saved ten input features. The matched control, its hybrid, the standalone audio-aware model, and its hybrid remain separate columns. The earlier AMI candidate and the audio-aware candidate are research artifacts. The historical official Whispered test CSV in the repository is from the logistic baseline and is outside this handoff.

## Verify from a normalized CSV

`true_high_error` is one only if `wer > 0.20`. The archived evaluation uses the published WER. The live evaluations use WER from fresh outputs with case, punctuation, and whitespace normalization. `classifier_risk` is the original model’s positive-class score. `native_risk` equals `-avg_logprob` with missing confidence ranked first. In the CSV missing confidence is blank and flagged by `native_confidence_missing`. `original_hybrid_risk` is the mean of the two percentile ranks computed across each complete evaluation file. Percentiles use average ties. `recording_group` is a linked meeting family for AMI, a chapter ID for LibriSpeech, and a speaker ID for Common Voice. Use LibriSpeech’s `speaker_id` for uncertainty grouped by reader.

To reproduce a review list, sort the method’s risk descending and `clip_id` ascending. The original Common Voice, LibriSpeech and first AMI experiments rounded `0.20 × N` up. The latest audio-aware experiment rounded down when needed. With 640 clips its budget was exactly 128. Calculate precision as catches over review slots and recall as catches over all high-error clips.

```python
from math import ceil
import pandas as pd

clips = pd.read_csv("common_voice_dev_verification.csv", float_precision="round_trip")
native = -clips["avg_logprob"].fillna(float("-inf"))
clips["recomputed_hybrid"] = (
    native.rank(method="average", pct=True)
    + clips["classifier_risk"].rank(method="average", pct=True)
) / 2
budget = ceil(0.20 * len(clips))
top = clips.sort_values(["recomputed_hybrid", "clip_id"], ascending=[False, True]).head(budget)
assert set(top.clip_id) == set(clips.loc[clips.original_hybrid_review, "clip_id"])
caught = int(top.true_high_error.sum())
print(caught, caught / budget, caught / clips.true_high_error.sum())
```

For a source-level rebuild, run the following from the project root with scikit-learn 1.8.0, joblib, pandas, NumPy and PyArrow. `PATH_TO_RESULT_ZIPS` is a folder containing the five archives named in `provenance.json`. `PATH_TO_ENGLISH_DEV_PARQUET` is `common_voice_17/en/dev/dev-00000.parquet` at [the pinned Whispered revision](https://huggingface.co/datasets/burakaydinofficial/Whispered/tree/25299fd560dde4271f5177118354af5a961b92fd/common_voice_17/en/dev). The exporter checks ranking and budget against the source summaries and saved review memberships when present.

```shell
python -m src.export_evaluation_handoff --source PATH_TO_RESULT_ZIPS --development-metadata PATH_TO_ENGLISH_DEV_PARQUET --output tmp/evaluation_handoff
```

Member C can independently reproduce the table using these records and their existing evaluation implementation. Keep the four cohorts separate, report the original rounding rules, and use grouped uncertainty if reporting a gain. LibriSpeech has just 22 positives and the first AMI holdout has four independent groups. The explored Common Voice cohort is development evidence. The later AMI holdout is fresh within the same meeting corpus and domain. Small improvements in individual cohorts do not establish a reliable advantage over native confidence.
