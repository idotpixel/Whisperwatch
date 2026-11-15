# Run WHISPERWATCH

Use Python 3.12 or 3.13 for the demo. From the project root:

```bash
python -m pip install "gradio==5.50.0"
python app.py
```

Open the URL Gradio prints and select **Open Workspace**. Dataset Replay and Model Evaluation use the packaged files in `assets/verified/`. The live tools need Member A’s backend and verified classifier artifact.

On Gradio 4, the landing and workspace share a page. A compatible frontend-only environment uses Python 3.12 with `gradio==4.44.1`, `huggingface-hub<1`, `pydantic==2.10.6`, and `fastapi==0.115.12`. Use Gradio 5 for the live backend, whose handoff pins Hugging Face Hub 1.32.0.

## Google Colab with the live models

1. Start Member A’s backend Colab notebook in a T4 GPU runtime. Upload the backend and verified classifier bundles when it asks.
2. Make sure the integrated project directory contains `app.py`, `ui/`, `assets/`, Member A’s `src/` files, and the installed model under `models/`.
3. In a Colab cell, run `%pip install "gradio==5.50.0"`. Keep the dependencies pinned by the backend handoff.
4. Run the following from the notebook. Set `project` to the integrated folder if it differs from the backend notebook’s default.

```python
import subprocess
import sys
from src.colab_backend import gpu_process_env

subprocess.run(
    [sys.executable, "-u", "app.py", "--share"],
    cwd=project,
    env=gpu_process_env(),
    check=True,
)
```

Open the Gradio share URL printed in the cell. As a standalone entry point, `python app.py` also enables sharing when it detects a Colab runtime.

## Read the results

- **Live:** WHISPERWATCH displays the classifier’s 0–1 score, Whisper’s mean segment log probability, and advice based on the demo’s 0.70 cutoff. Batch ranking uses the backend’s cohort-relative top-20% budget. Both displays help prioritize human review.
- **Replay:** The 50 archived cases display the saved hybrid priority and budget decision, classifier risk, Whisper transcription, human reference, and measured WER. Common Voice cases come from the development set.
- **Benchmarks:** The table uses all 18 rows in Member C’s `verified_experiments.json`. Its F1 column is derived from verified counts. The four PNGs are the original charts from Member C’s frontend handoff. The budget sweep CSV is included alongside them. The summary names the two holdouts with explicit random baselines.

The app looks for Member C’s files in `team_handoff/for_member_b_frontend_ui/`, then `reports/`, then `assets/verified/`. You can also specify `--replay PATH` or `--evaluation PATH`. Keep `figures/` and `dataset_limitations.json` beside a custom `verified_experiments.json` file.
