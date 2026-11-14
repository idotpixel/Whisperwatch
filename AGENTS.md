# WhisperWatch — Agent Instructions

## Project Overview

WhisperWatch is a machine-learning application developed for the Veritas AI International High School Hackathon 2026.

**Track:** AI + Languages

WhisperWatch evaluates AI-generated speech transcriptions and predicts which transcripts are most likely to require human review.

The system uses two AI models:

1. **OpenAI Whisper Large V3** — A pretrained speech-to-text model that generates transcriptions and confidence-related measurements.
2. **WhisperWatch** — Our custom-trained machine-learning classifier that predicts whether a transcription is likely to contain significant errors.

The system does NOT automatically correct transcription errors.

## Technology Stack

- Python
- Google Colab (primary execution environment)
- GitHub (source control)
- Hugging Face Datasets
- Faster-Whisper (Whisper Large V3)
- scikit-learn
- Gradio
- pandas and NumPy
- jiwer
- joblib

The application must work in Google Colab.

## Team Roles

### Member A — Technical Lead (Codex 1)

**Branch:** `feature/model-asr`

Responsible for:
- Dataset loading and preprocessing
- Feature extraction
- ML model training
- Whisper integration
- Model serialization
- Final application integration
- Reviewing and merging pull requests

Primary files:
- `src/features.py`
- `src/train.py`
- `src/inference.py`
- `src/asr.py`
- `notebooks/`
- `requirements.txt` (coordinate shared dependency changes)

### Member B — UI/UX Lead (Codex 2)

**Branch:** `feature/ui`

Responsible for:
- Application design
- Gradio interface
- Audio upload experience
- Transcript display
- Prediction visualization
- User experience and accessibility

Primary files:
- `app.py`
- `ui/`
- `assets/`

### Member C — Evaluation Lead (Antigravity)

**Branch:** `feature/evaluation`

Responsible for:
- Baseline implementation
- Model evaluation
- Precision, recall, F1, and confusion matrices
- Review-prioritization experiments
- Error analysis
- Evaluation tests and reports

Primary files:
- `src/evaluate.py`
- `tests/`
- `reports/`

## Collaboration Rules

1. Work exclusively on your assigned branch.
2. Only modify files belonging to your role unless the technical lead approves otherwise.
3. Never commit directly to `main`.
4. Create a pull request when completing work.
5. Do not merge without technical lead approval.
6. Pull the latest `main` branch before starting a new task when needed.
7. Do not overwrite or delete another agent's work.
8. Do not introduce unnecessary dependencies.
9. Keep functions modular and documented.
10. Ensure code is compatible with Google Colab.
11. Verify the correct branch before making changes.
12. Do not automatically commit, push, or merge without the member's approval.

## Shared Integration Contracts

### Feature Extraction

File: `src/features.py`

Function:

`extract_features(segments, transcript: str, duration_s: float) -> dict`

Extract numerical features consistently during both training and live inference.

### Speech Recognition

File: `src/asr.py`

Function:

`transcribe_audio(audio_path: str) -> dict`

Expected output:
- `transcript`
- `segments`
- `duration_s`

### Complete Inference

File: `src/inference.py`

Function:

`analyze_audio(audio_path: str) -> dict`

Expected output:
- `transcript`
- `review_score`
- `review_label`
- `details`

Higher `review_score` values indicate greater predicted transcription-error risk.

### Evaluation

File: `src/evaluate.py`

Expected prediction-data fields:
- `clip_id`
- `language`
- `true_high_error`
- `risk_score`
- `avg_logprob`

The evaluation module must accept model outputs without requiring changes to the training pipeline.

## Machine Learning Requirements

The custom model MUST be trained on actual labeled examples.

Use the Whispered dataset where suitable:

https://huggingface.co/datasets/burakaydinofficial/Whispered

Training labels are derived from the difference between human reference transcriptions and Whisper-generated transcriptions.

**Initial high-error threshold:** WER > 0.20.

NEVER use the following as model input features:
- Human reference transcripts
- True WER values
- Ground-truth error labels
- Features calculated from reference transcripts
- Reference-based forced alignments

These are permitted only for training-label generation and evaluation.

Use only features that will also be available during real application inference.

Avoid data leakage.

Use the appropriate dataset splits:
- `train`: Training
- `dev`: Validation and model selection
- `test`: Final held-out evaluation
- `other`: Excluded due to documented overlap

Do not mix outputs from incompatible Whisper model versions without explicitly validating compatibility.

## Evaluation Requirements

Compare the trained classifier against:
- A simple baseline
- Whisper's native confidence-related signals

Report:
- Precision
- Recall
- F1 score
- Confusion matrix
- Review-prioritization performance
- Examples of incorrect predictions

Use held-out test data for final evaluation.

Never fabricate results or performance claims.

## Application Requirements

### Required MVP — Dataset Replay Mode

The application must:
1. Load real, previously evaluated transcription examples.
2. Display the Whisper-generated transcript.
3. Run or display the actual trained classifier's prediction.
4. Display an estimated review-risk score.
5. Recommend human review when appropriate.
6. Allow users to reveal the human reference transcript.
7. Display genuine evaluation results.

### Optional Upgrade — Live Audio Analysis

If sufficient time and resources are available, the application should:

1. Accept an audio recording.
2. Transcribe it using pretrained Whisper Large V3.
3. Extract the required inference features.
4. Run the trained WhisperWatch classifier.
5. Display the transcription.
6. Display an estimated review-risk score.
7. Recommend human review when appropriate.

Historical examples may display the correct reference transcript for comparison.

For arbitrary uploads, the correct transcript is not automatically available.

## Development Principles

- Prioritize a working minimum viable product.
- Do not add features without approval.
- Never hardcode fake model predictions.
- Never commit API keys or credentials.
- Keep large datasets outside Git.
- Keep code understandable to beginners.
- Document assumptions and limitations.
- Explain how each component works.
- Use actual model outputs for demonstrations.
- Do not claim that an uncalibrated risk score represents a guaranteed error probability.

## Definition of Done

A feature is complete when:
- Its code runs successfully.
- Its inputs and outputs are documented.
- Relevant tests pass.
- It integrates with the agreed interfaces.
- It has been committed to its feature branch.
- A pull request is ready for review.

## Hackathon Deadline

**November 16, 2026 — 8:59 AM UAE time**

Prioritize working code, reproducible evaluation, and a complete submission over unnecessary features.
