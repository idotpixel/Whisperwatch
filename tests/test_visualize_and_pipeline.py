"""Tests for visualization module and end-to-end evaluation pipeline."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.visualize import plt
from src.visualize import (
    plot_confusion_matrix,
    plot_baseline_comparison,
    plot_review_budget_curve,
)
from run_evaluation import create_synthetic_demo_dataset


def test_visualizations_generate_valid_files():
    """Verify that plotting functions generate valid PNG files without GUI display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Confusion matrix plot
        cm_dict = {"tp": 25, "fp": 5, "tn": 65, "fn": 5, "total": 100}
        cm_file = plot_confusion_matrix(cm_dict, tmp_path / "cm.png")
        assert cm_file.exists()
        assert cm_file.stat().st_size > 1000

        # 2. Baseline comparison plot
        comp_df = pd.DataFrame([
            {"method": "Random Review", "recall_at_budget": 0.20, "precision_at_budget": 0.15},
            {"method": "Whisper Native", "recall_at_budget": 0.55, "precision_at_budget": 0.40},
            {"method": "WhisperWatch", "recall_at_budget": 0.85, "precision_at_budget": 0.70},
        ])
        comp_file = plot_baseline_comparison(comp_df, tmp_path / "comp.png")
        assert comp_file.exists()
        assert comp_file.stat().st_size > 1000

        # 3. Budget curve plot
        curve_df = pd.DataFrame([
            {"budget": 0.10, "method": "Random", "recall_at_budget": 0.10},
            {"budget": 0.20, "method": "Random", "recall_at_budget": 0.20},
            {"budget": 0.10, "method": "WhisperWatch", "recall_at_budget": 0.50},
            {"budget": 0.20, "method": "WhisperWatch", "recall_at_budget": 0.80},
        ])
        curve_file = plot_review_budget_curve(curve_df, tmp_path / "curve.png")
        assert curve_file.exists()
        assert curve_file.stat().st_size > 1000


def test_create_synthetic_demo_dataset():
    """Verify synthetic fixture creation adheres to dataset specification."""
    df = create_synthetic_demo_dataset(n_samples=50, random_seed=123)

    assert len(df) == 50
    assert "clip_id" in df.columns
    assert "language" in df.columns
    assert "wer" in df.columns
    assert "true_high_error" in df.columns
    assert "avg_logprob" in df.columns
    assert "risk_score" in df.columns
    assert "whisper_text" in df.columns
    assert "human_text" in df.columns

    # Check high error relationship
    assert (df["true_high_error"] == (df["wer"] > 0.20).astype(int)).all()


def test_baseline_chart_preserves_distinct_verified_method_labels(tmp_path, monkeypatch):
    """The archived comparison exports use lowercase method identifiers."""
    frame = pd.DataFrame({
        "method": ["random", "classifier", "native", "original_hybrid"],
        "recall_at_budget": [0.20, 0.33, 0.35, 0.36],
        "precision_at_budget": [0.35, 0.57, 0.59, 0.61],
    })
    try:
        with monkeypatch.context() as patch:
            patch.setattr(plt, "close", lambda *args: None)
            plot_baseline_comparison(frame, tmp_path / "labels.png")
            labels = [tick.get_text().splitlines()[0] for tick in plt.gca().get_xticklabels()]
    finally:
        plt.close("all")
    assert labels == ["Random Review", "WhisperWatch", "Whisper Confidence", "WhisperWatch Hybrid"]
