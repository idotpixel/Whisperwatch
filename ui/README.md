# Run WhisperWatch

## Live audio uploads

Use Python 3.12 from the repository folder.

```bash
python -m pip install -r requirements-live.txt
python -m src.setup_demo "/path/to/model.zip"
python app.py
```

Use the approved classifier ZIP supplied by the team. The setup command verifies the trained classifier before installing it under `models/`. Whisper weights are downloaded on the first live request. Open the address printed by Gradio and select **Open Workspace**. Upload an English recording up to 30 seconds. CPU inference can take a minute or more.

## Google Colab with the live models

Open [the live demo notebook](https://colab.research.google.com/github/idotpixel/Whisperwatch/blob/main/notebooks/backend_colab.ipynb), choose **Runtime → Change runtime type → T4 GPU**, and run the cells. The notebook clones the public code and asks the host to upload the classifier ZIP. It verifies the model and prints a Gradio link that visitors can use while the Colab runtime is active.

## Replay and benchmarks only

To browse the historical cases and evaluation charts without installing the speech backend, run:

```bash
python -m pip install "gradio==5.50.0"
python app.py
```

## Read the results

- **Live:** WHISPERWATCH displays the classifier’s 0–1 score, Whisper’s mean segment log probability, and advice based on the demo’s 0.70 cutoff. Batch ranking uses the backend’s cohort-relative top-20% budget. Both displays help prioritize human review.
- **Replay:** Seven handpicked historical cases display both saved review decisions at the original 20% budget, the hybrid priority, classifier risk, Whisper transcription, human reference, and measured WER. Common Voice cases come from the development set.
- **Benchmarks:** The table uses all 18 rows in Member C’s `verified_experiments.json`. Its F1 column is derived from verified counts. The four PNGs are the original charts from Member C’s frontend handoff. The budget sweep CSV is included alongside them. The summary names the two holdouts with explicit random baselines.

The benchmark tab uses the verified assets packaged in `assets/verified/`. If they are missing, it looks in `team_handoff/for_member_b_frontend_ui/` and then `reports/`. Use `--replay PATH` for a different replay export. Use `--evaluation PATH` to override the benchmark data. Keep `figures/` and `dataset_limitations.json` beside a custom `verified_experiments.json` file.

## Verify the historical cases

The showcase prioritizes transcript pairs with visible changes to content words, timing, or omitted information. It also includes a zero-WER false positive and cases where the hybrid and native-confidence queues disagree in each direction. The pairs were read against their references before selection. The cases illustrate individual outcomes. Use the benchmark tab for performance across cohorts.

`assets/replay_cases.json` records the evaluation commit, source CSV for each cohort, and original clip IDs. The case titles and notes are editorial. Run the verifier from the project root with a checkout that contains the pinned evaluation commit. If needed, fetch the evaluation branch with `git fetch origin feature/evaluation`.

```bash
python -m ui.verify_replay
```

The verifier checks both methods against the saved review flags and published counts across AMI (150 of 750), Common Voice 17 development (1,631 of 8,152), and LibriSpeech (100 of 499). It uses the original full-cohort average-tie percentiles and descending score with ascending clip ID as the tie-breaker. It also checks each curated record’s exact transcripts, WER, label, scores, review flags, and displayed outcome. The app reads the saved values directly.
