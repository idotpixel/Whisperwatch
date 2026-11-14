"""Automated tests for WhisperWatch Evaluation Module (src/evaluate.py).

Covers all required edge cases and behaviors:
- Correct precision, recall, F1, and prevalence calculations.
- Confusion matrix correctness across all quadrants (TP, FP, TN, FN).
- Correct risk-score ranking (descending: high risk first).
- Correct native Whisper log-probability direction handling (ascending: low logprob first).
- Correct top-20% review budget selection and lift computation.
- Graceful handling of missing values (NaNs).
- Graceful handling of empty inputs.
- Zero-positive scenarios (no high-error examples in data).
- All-positive scenarios (all clips are high-error).
- Prediction data loading, deduplication, language filtering, and schema validation.
- Frontend example export compatibility.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.evaluate import (
    compute_prevalence,
    compute_confusion_matrix,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_average_precision,
    compute_roc_auc,
    evaluate_classification_metrics,
    evaluate_review_budget,
    compare_baselines,
    evaluate_budget_curve,
    load_predictions,
    extract_error_analysis_cases,
    export_frontend_examples,
    DEFAULT_WER_THRESHOLD,
    DEFAULT_REVIEW_BUDGET,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_synthetic_predictions():
    """Generates a clean synthetic dataset with 20 clips and known WERs.

    Prevalence: 5 high-error clips (WER > 0.20) out of 20 = 25%.
    """
    data = {
        "clip_id": [f"clip_{i:03d}" for i in range(20)],
        "language": ["en"] * 18 + ["es", "fr"],  # 2 non-English clips
        "wer": [
            0.00, 0.05, 0.08, 0.12, 0.15,  # 5 clean clips
            0.02, 0.04, 0.10, 0.18, 0.19,  # 5 clean clips
            0.01, 0.03, 0.07, 0.14, 0.16,  # 5 clean clips
            0.45, 0.55, 0.70, 0.85, 0.90,  # 5 high-error clips (WER > 0.20)
        ],
        # Whisper average logprob: lower means less confident
        "avg_logprob": [
            -0.10, -0.15, -0.20, -0.25, -0.30,
            -0.12, -0.18, -0.22, -0.28, -0.35,
            -0.11, -0.14, -0.19, -0.26, -0.32,
            -0.80, -0.90, -1.20, -1.50, -2.00,  # High error has lower logprob
        ],
        # WhisperWatch risk score: higher means higher error risk
        "risk_score": [
            0.05, 0.10, 0.12, 0.18, 0.22,
            0.08, 0.11, 0.15, 0.25, 0.28,
            0.06, 0.09, 0.14, 0.20, 0.24,
            0.75, 0.82, 0.88, 0.94, 0.98,  # High error has high risk score
        ],
        "whisper_text": [f"Whisper transcript {i}" for i in range(20)],
        "human_text": [f"Human transcript {i}" for i in range(20)],
    }
    df = pd.DataFrame(data)
    df["true_high_error"] = (df["wer"] > 0.20).astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. Classification Metrics Tests
# ---------------------------------------------------------------------------

def test_compute_prevalence():
    """Verify prevalence calculates positive class proportion correctly."""
    assert compute_prevalence([0, 0, 0, 0]) == 0.0
    assert compute_prevalence([1, 1, 1, 1]) == 1.0
    assert compute_prevalence([1, 0, 0, 0, 0]) == 0.2
    assert compute_prevalence([]) == 0.0
    assert compute_prevalence(np.array([1, 1, 0, 0])) == 0.5


def test_compute_confusion_matrix():
    """Verify confusion matrix counts across TP, FP, TN, FN."""
    # 2 TP, 1 FP, 2 TN, 1 FN
    y_true = [1, 1, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 0, 0, 0]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["tn"] == 2
    assert cm["fn"] == 1
    assert cm["total"] == 6


def test_confusion_matrix_empty_and_mismatch():
    """Verify empty input and shape mismatch behavior."""
    cm = compute_confusion_matrix([], [])
    assert cm["total"] == 0
    assert cm["tp"] == 0

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_confusion_matrix([1, 0], [1])


def test_precision_recall_f1():
    """Verify precision, recall, and F1 calculations against manual formulas."""
    # TP=2, FP=1, FN=1 -> Prec = 2/3, Rec = 2/3, F1 = 2/3
    y_true = [1, 1, 0, 1]
    y_pred = [1, 1, 1, 0]

    assert pytest.approx(compute_precision(y_true, y_pred), 0.001) == 2 / 3
    assert pytest.approx(compute_recall(y_true, y_pred), 0.001) == 2 / 3
    assert pytest.approx(compute_f1(y_true, y_pred), 0.001) == 2 / 3


def test_metrics_zero_division():
    """Verify zero division returns 0.0 safely instead of throwing errors."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    assert compute_precision(y_true, y_pred, zero_division=0.0) == 0.0
    assert compute_recall(y_true, y_pred, zero_division=0.0) == 0.0
    assert compute_f1(y_true, y_pred, zero_division=0.0) == 0.0


def test_average_precision_and_roc_auc():
    """Verify PR-AUC and ROC-AUC for perfect vs inverted rankings."""
    y_true = [0, 0, 1, 1]
    perfect_scores = [0.1, 0.2, 0.8, 0.9]
    inverted_scores = [0.9, 0.8, 0.2, 0.1]

    # Perfect ranking
    assert compute_average_precision(y_true, perfect_scores) == 1.0
    assert compute_roc_auc(y_true, perfect_scores) == 1.0

    # Inverted ranking
    assert compute_roc_auc(y_true, inverted_scores) == 0.0

    # Single class edge cases
    assert compute_average_precision([0, 0], [0.5, 0.6]) == 0.0
    assert compute_roc_auc([0, 0], [0.5, 0.6]) is None


def test_evaluate_classification_metrics_summary():
    """Verify all metrics bundle into a complete report dictionary."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    scores = [0.9, 0.1, 0.4, 0.2]

    res = evaluate_classification_metrics(y_true, y_pred, scores=scores)
    assert res["prevalence"] == 0.5
    assert res["tp"] == 1
    assert res["fn"] == 1
    assert res["fp"] == 0
    assert res["tn"] == 2
    assert "pr_auc" in res
    assert "roc_auc" in res


# ---------------------------------------------------------------------------
# 2. Review Budget & Direction Handling Tests
# ---------------------------------------------------------------------------

def test_top_20_percent_selection(clean_synthetic_predictions):
    """Verify that a 20% budget on 20 clips selects exactly 4 clips."""
    df = clean_synthetic_predictions
    res = evaluate_review_budget(df, priority_col="risk_score", budget=0.20, ascending=False)

    assert res["total_clips"] == 20
    assert res["reviewed_clips"] == 4
    assert res["budget_fraction"] == 0.20


def test_risk_score_direction_descending():
    """Verify that risk_score ranks higher values first."""
    df = pd.DataFrame({
        "risk_score": [0.1, 0.9, 0.3, 0.8],
        "true_high_error": [0, 1, 0, 1],
    })
    # 50% budget -> top 2 clips (should pick 0.9 and 0.8, both high error)
    res = evaluate_review_budget(df, priority_col="risk_score", budget=0.50, ascending=False)
    assert res["reviewed_clips"] == 2
    assert res["high_error_caught"] == 2
    assert res["recall_at_budget"] == 1.0
    assert res["precision_at_budget"] == 1.0


def test_native_logprob_direction_ascending():
    """Verify that avg_logprob ranks lower (more negative) values first.

    In Whisper: -2.0 is lower confidence than -0.1. A reviewer targeting
    errors must inspect the least confident clips first.
    """
    df = pd.DataFrame({
        "avg_logprob": [-0.05, -2.50, -0.10, -1.80],  # -2.50 and -1.80 are least confident
        "true_high_error": [0, 1, 0, 1],
    })
    # Top 50% review budget using ascending=True
    res = evaluate_review_budget(df, priority_col="avg_logprob", budget=0.50, ascending=True)

    assert res["reviewed_clips"] == 2
    assert res["high_error_caught"] == 2
    assert res["recall_at_budget"] == 1.0

    # Inverting ascending flag would erroneously review confident clips first
    bad_res = evaluate_review_budget(df, priority_col="avg_logprob", budget=0.50, ascending=False)
    assert bad_res["high_error_caught"] == 0
    assert bad_res["recall_at_budget"] == 0.0


def test_lift_over_random():
    """Verify lift formula: lift = recall / budget."""
    df = pd.DataFrame({
        "risk_score": [0.1, 0.2, 0.8, 0.9],
        "true_high_error": [0, 0, 1, 1],
    })
    # Budget 50%: top 2 selected out of 4, both are positive -> recall = 1.0
    # Lift = 1.0 / 0.50 = 2.0x
    res = evaluate_review_budget(df, priority_col="risk_score", budget=0.50, ascending=False)
    assert res["recall_at_budget"] == 1.0
    assert res["lift_over_random"] == 2.0


# ---------------------------------------------------------------------------
# 3. Multi-Baseline Comparison Tests
# ---------------------------------------------------------------------------

def test_compare_baselines_structure(clean_synthetic_predictions):
    """Verify compare_baselines returns all 3 methods on identical records."""
    df = clean_synthetic_predictions
    comp = compare_baselines(df, budget=0.20)

    assert len(comp) == 3
    methods = list(comp["method"])
    assert any("Random" in m for m in methods)
    assert any("Native Confidence" in m for m in methods)
    assert any("WhisperWatch" in m for m in methods)

    # Check columns
    assert "recall_at_budget" in comp.columns
    assert "precision_at_budget" in comp.columns
    assert "lift_over_random" in comp.columns
    assert "relative_improvement_vs_whisper" in comp.columns

    # Verify identical record count evaluated
    for clips in comp["clips_reviewed"]:
        assert clips == 4  # 20% of 20 = 4


def test_compare_baselines_whisperwatch_improvement():
    """Verify scenario where WhisperWatch identifies errors Whisper missed."""
    # Whisper hallucinated: gave high confidence (-0.05) to a clip with WER=0.80
    df = pd.DataFrame({
        "clip_id": ["c1", "c2", "c3", "c4", "c5"],
        "wer": [0.80, 0.05, 0.02, 0.01, 0.03],
        "true_high_error": [1, 0, 0, 0, 0],
        # Whisper confidence failed: c1 has highest logprob (appears confident)
        "avg_logprob": [-0.05, -1.20, -1.10, -0.90, -0.80],
        # WhisperWatch caught it: c1 has highest risk score
        "risk_score": [0.95, 0.20, 0.15, 0.10, 0.05],
    })

    comp = compare_baselines(df, budget=0.20)
    whisper_row = comp[comp["method"].str.contains("Native Confidence")].iloc[0]
    ww_row = comp[comp["method"].str.contains("WhisperWatch")].iloc[0]

    # At 20% budget (1 clip reviewed):
    # Whisper reviews c2 (logprob -1.20) -> WER 0.05 -> Caught 0
    assert whisper_row["high_error_caught"] == 0
    assert whisper_row["recall_at_budget"] == 0.0

    # WhisperWatch reviews c1 (risk 0.95) -> WER 0.80 -> Caught 1
    assert ww_row["high_error_caught"] == 1
    assert ww_row["recall_at_budget"] == 1.0


def test_evaluate_budget_curve(clean_synthetic_predictions):
    """Verify budget curve sweeps over multiple budgets correctly."""
    df = clean_synthetic_predictions
    budgets = [0.10, 0.20, 0.50]
    curve = evaluate_budget_curve(df, budgets=budgets)

    assert len(curve) == 3 * len(budgets)  # 3 methods * 3 budgets = 9 rows
    assert set(curve["budget"].unique()) == set(budgets)


# ---------------------------------------------------------------------------
# 4. Edge Cases: Missing Values, Empty Input, Zero/All High-Error
# ---------------------------------------------------------------------------

def test_handling_of_missing_values():
    """Verify evaluation handles NaNs in columns gracefully."""
    df = pd.DataFrame({
        "risk_score": [0.9, np.nan, 0.1, 0.8],
        "true_high_error": [1, 1, 0, np.nan],
    })
    # Should evaluate only the 2 completely valid rows (index 0 and 2)
    res = evaluate_review_budget(df, priority_col="risk_score", budget=0.50)
    assert res["total_clips"] == 2
    assert res["reviewed_clips"] == 1


def test_empty_input_behavior():
    """Verify empty DataFrame does not raise unhandled exceptions."""
    empty_df = pd.DataFrame(columns=["risk_score", "avg_logprob", "true_high_error"])

    res = evaluate_review_budget(empty_df, priority_col="risk_score")
    assert res["total_clips"] == 0
    assert res["reviewed_clips"] == 0
    assert res["recall_at_budget"] == 0.0

    comp = compare_baselines(empty_df)
    assert comp.empty


def test_no_high_error_examples():
    """Verify behavior when no clips have high error (prevalence = 0%)."""
    df = pd.DataFrame({
        "risk_score": [0.1, 0.2, 0.3, 0.4],
        "avg_logprob": [-0.1, -0.2, -0.3, -0.4],
        "true_high_error": [0, 0, 0, 0],
    })

    comp = compare_baselines(df, budget=0.25)
    for _, row in comp.iterrows():
        assert row["total_high_error"] == 0
        assert row["high_error_caught"] == 0
        assert row["recall_at_budget"] == 0.0
        assert row["precision_at_budget"] == 0.0


def test_all_high_error_examples():
    """Verify behavior when every clip is high-error (prevalence = 100%)."""
    df = pd.DataFrame({
        "risk_score": [0.9, 0.8, 0.7, 0.6],
        "avg_logprob": [-1.0, -1.2, -1.5, -1.8],
        "true_high_error": [1, 1, 1, 1],
    })

    comp = compare_baselines(df, budget=0.50)
    for _, row in comp.iterrows():
        assert row["total_high_error"] == 4
        assert row["precision_at_budget"] == 1.0


# ---------------------------------------------------------------------------
# 5. Prediction Data Loading & Integrity Tests
# ---------------------------------------------------------------------------

def test_load_predictions_csv_and_derivation():
    """Verify loading from CSV, auto-deriving true_high_error from wer, and filtering."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("clip_id,language,wer,avg_logprob,risk_score\n")
        f.write("c1,en,0.25,-1.5,0.85\n")  # High error (WER > 0.20)
        f.write("c2,en,0.05,-0.2,0.15\n")  # Low error
        f.write("c3,es,0.30,-1.8,0.90\n")  # Non-English
        f.write("c1,en,0.25,-1.5,0.85\n")  # Duplicate clip_id
        temp_path = f.name

    try:
        # Load with English filter and deduplication
        df, meta = load_predictions(temp_path, filter_language="en", check_duplicates=True)
        assert len(df) == 2  # c1 and c2 only (c3 filtered by language, duplicate c1 removed)
        assert meta["language_filtered"] == "en"
        assert meta["duplicate_clip_ids_removed"] == 1
        assert "true_high_error" in df.columns
        assert df.loc[df["clip_id"] == "c1", "true_high_error"].iloc[0] == 1
        assert df.loc[df["clip_id"] == "c2", "true_high_error"].iloc[0] == 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_predictions_missing_required_column():
    """Verify clear KeyError when a vital predictor column is absent."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("clip_id,avg_logprob\n")  # Missing risk_score
        f.write("c1,-1.5\n")
        temp_path = f.name

    try:
        with pytest.raises(KeyError, match="Missing required prediction column"):
            load_predictions(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. Error Analysis & Frontend Export Tests
# ---------------------------------------------------------------------------

def test_extract_error_analysis_cases(clean_synthetic_predictions):
    """Verify pedagogical error categories are extracted cleanly."""
    df = clean_synthetic_predictions
    cases = extract_error_analysis_cases(df, budget=0.20, n_samples=3)

    assert "whisperwatch_wins" in cases
    assert "false_alarms_avoided" in cases
    assert "whisperwatch_false_alarms" in cases
    assert "missed_errors" in cases


def test_export_frontend_examples(clean_synthetic_predictions):
    """Verify frontend export generates valid JSON with expected schema for Member B."""
    df = clean_synthetic_predictions
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "demo_examples.json"
        res_path = export_frontend_examples(df, out_path, budget=0.20, n_examples=10)

        assert res_path.exists()
        import json
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0
        first = data[0]
        assert "clip_id" in first
        assert "whisper_text" in first
        assert "human_text" in first
        assert "risk_score" in first
        assert "recommend_review" in first
        assert "outcome_category" in first
