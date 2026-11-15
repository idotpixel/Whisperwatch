# WhisperWatch backend handoff for Member B

WhisperWatch helps subtitle editors choose what to check first. The user uploads a short English recording. Whisper Large V3 writes the transcript and supplies segment measurements. Our frozen classifier uses ten measurements from that output and the recording duration to score review risk. With several recordings, the backend can put the clips in review order.

## Connect the backend

Run `notebooks/backend_colab.ipynb` in a hosted T4 GPU runtime. The notebook installs the backend from GitHub and verifies the classifier from the model-only release. For local setup, install `requirements-live.txt` and run `python -m src.setup_demo` from the repository root.

The live backend uses Python, pandas 2.2.3, scikit-learn 1.8.0, joblib 1.5.3, faster-whisper 1.2.1, CTranslate2 4.9.0, PyAV 16.0.1 and Hugging Face Hub 1.32.0. The T4 path also needs CUDA 12 cuBLAS and cuDNN 9. The first live request downloads and caches about 3.1 GB of pinned Large V3 weights. Gradio is part of the UI environment. Install it using Member B’s frontend setup when integrated. [Large V3 weights](https://huggingface.co/Systran/faster-whisper-large-v3) · [CUDA requirements](https://github.com/SYSTRAN/faster-whisper#gpu)

Run the finished app from the **integrated checkout’s root**, with the verified model under `models/`. On Colab, start it in a Python process with the CUDA environment set before Python starts. The backend notebook exercises this same process setup without launching the UI.

```python
# After integration, from the Colab notebook containing the project code.
import subprocess
import sys
from src.colab_backend import gpu_process_env

# Use the integrated checkout path and the actual frontend entry point.
process = subprocess.Popen([sys.executable, "app.py"], cwd=project, env=gpu_process_env())
```

## Call the backend from Gradio

Use a Gradio audio or file input that passes a **filepath**. Allow English clips longer than zero and at most 30 seconds. An empty upload, unreadable file, unsupported clip duration, or missing runtime dependency can raise an exception. Show a friendly error in the callback when that happens.

```python
from src.asr import analyze_audio
from src.inference import rank_recordings

def single_callback(audio_path):
    result = analyze_audio(audio_path)
    return result["transcript"], result["review_score"], result["review_label"], result["details"]

def batch_callback(audio_paths, method="hybrid"):
    analyzed = [dict(analyze_audio(path), clip_id=f"clip-{i:04d}")
                for i, path in enumerate(audio_paths, 1)]
    return rank_recordings(analyzed, method=method)
```

| Function | Input | Return value |
|---|---|---|
| `src.asr.transcribe_audio` | Path to an audio file | `transcript`, a list of faster-whisper `segments`, and `duration_s` |
| `src.features.extract_features` | Segments, generated transcript, duration | Ten ordered numeric features. Missing measurements are `NaN` internally |
| `src.inference.predict_from_features` | A feature dictionary | `review_score`, `review_label`, and JSON-safe `details` |
| `src.asr.analyze_audio` | Path to a short English audio file | The prediction fields plus `transcript`. Use this for individual uploads |
| `src.inference.rank_recordings` | Iterable of analysis dictionaries, each with a unique string `clip_id`, plus `method` | A sorted list with `clip_id`, `transcript`, `classifier_risk`, `native_confidence_risk`, `hybrid_risk`, `ranking_method`, 1-based `rank`, and `selected_for_review` |
| `src.inference.install_production_model` | Trusted original results ZIP path or file-like object | Path to the checksum-verified model |
| `src.inference.load_production_model` | Optional verified model path | The cached frozen artifact and its schema |

This is a genuine score reproduced from saved live LibriSpeech segments. The audio itself does not ship in the fixture. [LibriSpeech attribution](https://www.openslr.org/12/)

```text
Transcript: We were out of ammunition and were helpless had the fight been pressed.
Classifier review score: 0.018378917987081748
Review guidance: Compare in batch for review priority
```

## Label scores correctly in the UI

- **Classifier risk** is a model score from 0 to 1, with higher values indicating greater estimated error risk. Low scores can still miss errors. Avoid showing it as a calibrated probability or a verdict about transcript accuracy.
- **Native confidence risk** is negative mean Whisper segment log probability. Higher means more reason to review. Missing confidence is shown as `None` and gets highest native priority, per the frozen baseline rule.
- **Hybrid priority** averages the percentile ranks of the two risks across the supplied batch with 50/50 weights. It depends on the other clips in the batch. A fixed external reference distribution would require a separate design and validation. Each method selects the top `ceil(0.20 × n)`, with ties broken by ascending clip ID. For one upload, show the classifier score and guidance. Use a five-clip batch to demonstrate one review slot.
- **Actual WER** needs a human reference. It is unavailable for arbitrary uploads and plays no part in scoring. The backend helps prioritize human review. Editors still decide what to correct.

The CPU smoke tests validate the model and API using genuine saved segments and a stub decoder. The final notebook’s real-audio T4 step is the remaining integration smoke check. The frozen GPU configuration has run successfully in the earlier live evaluation notebooks.
