"""WhisperWatch Research Visualizations Module.

Generates presentation-ready evaluation charts and diagrams:
1. Confusion Matrix Heatmap (annotated with sample counts & percentages).
2. Multi-Baseline Comparison Bar Chart (Random vs Whisper Native vs WhisperWatch Classifier / Hybrid).
3. Review-Budget Effectiveness Curve (Recall vs Review Budget sweep from 5% to 50%).
4. Cross-Dataset Performance Summary (comparing Native Confidence vs Hybrid across all cohorts).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

# Configure headless matplotlib environment before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ww_mpl_config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set modern, readable aesthetic styles
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "WhisperWatch Error Detection — Confusion Matrix",
    model_name: str = "WhisperWatch Classifier",
) -> Path:
    """Generate an annotated confusion matrix visualization.

    Args:
        cm_dict: Dictionary with keys 'tp', 'fp', 'tn', 'fn', 'total'.
        output_path: File path to save the chart (.png).
        title: Plot title.
        model_name: Model identifier for caption.

    Returns:
        Path to saved PNG image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tp = cm_dict.get("tp", 0)
    fp = cm_dict.get("fp", 0)
    tn = cm_dict.get("tn", 0)
    fn = cm_dict.get("fn", 0)
    total = max(1, cm_dict.get("total", tp + fp + tn + fn))

    matrix = np.array([[tn, fp], [fn, tp]])
    matrix_pct = (matrix / total) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Annotations in each cell
    labels = [
        [f"True Negative\n(Accurate, Not Flagged)\n\n{tn} ({matrix_pct[0, 0]:.1f}%)",
         f"False Alarm\n(Accurate, Flagged)\n\n{fp} ({matrix_pct[0, 1]:.1f}%)"],
        [f"Missed Error\n(High Error, Not Flagged)\n\n{fn} ({matrix_pct[1, 0]:.1f}%)",
         f"True Positive\n(High Error, Flagged)\n\n{tp} ({matrix_pct[1, 1]:.1f}%)"],
    ]

    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(
                j, i, labels[i][j],
                ha="center", va="center", color=color,
                fontsize=10, weight="medium"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Flagged\n(Accept)", "Flagged for Review\n(Human Check)"])
    ax.set_yticklabels(["Acceptable\n(WER <= 20%)", "High Error\n(WER > 20%)"])
    ax.set_xlabel("WhisperWatch Action", labelpad=10, weight="bold")
    ax.set_ylabel("Ground Truth Quality", labelpad=10, weight="bold")
    ax.set_title(title, pad=15, weight="bold")

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    output_path: Union[str, Path],
    budget_pct: int = 20,
    title: Optional[str] = None,
) -> Path:
    """Generate a side-by-side bar chart comparing the prioritization methods.

    Compares:
    1. Random Review
    2. Whisper Native Confidence (avg_logprob)
    3. WhisperWatch Classifier (risk_score)
    4. WhisperWatch Hybrid (50/50 percentile blend, if present)

    Metrics shown:
    - Recall @ Budget (% of all errors captured)
    - Precision @ Budget (% of reviewed clips containing genuine errors)
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Prioritization Effectiveness at {budget_pct}% Review Budget"

    # Simplify method labels for visual clarity
    def clean_name(m: str) -> str:
        if "Random" in m:
            return "Random Review\n(Baseline 1)"
        if "Native" in m or ("Whisper" in m and "Watch" not in m and "Hybrid" not in m):
            return "Whisper Confidence\n(Baseline 2)"
        if "Hybrid" in m:
            return "WhisperWatch Hybrid\n(50/50 Percentile)"
        if "Audio-Aware" in m or "audio_aware" in m:
            return "Audio-Aware Model\n(Acoustic)"
        if "Control" in m:
            return "Control Model\n(10-Feature)"
        return "WhisperWatch\n(Classifier)"

    df = comparison_df.copy()
    labels = [clean_name(m) for m in df["method"]]
    recalls = [r * 100.0 for r in df["recall_at_budget"]]
    precisions = [p * 100.0 for p in df["precision_at_budget"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7.5, 2.0 * len(labels)), 5.2), dpi=150)
    rects1 = ax.bar(x - width / 2, recalls, width, label="Recall (% High Errors Caught)", color="#2563EB")
    rects2 = ax.bar(x + width / 2, precisions, width, label="Precision (% Reviewed Were Errors)", color="#059669")

    # Annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, weight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    ax.set_ylabel("Performance (%)", weight="bold")
    ax.set_title(title, pad=15, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight="medium")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(max(recalls + precisions, default=0) * 1.25, 40))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_review_budget_curve(
    curve_df: pd.DataFrame,
    output_path: Union[str, Path],
    target_budget: float = 0.20,
    title: str = "Review-Budget Effectiveness: High-Error Detection vs Review Capacity",
) -> Path:
    """Generate a review-budget effectiveness curve.

    Plots % of High Errors Caught (Recall) vs % of Clips Reviewed (Budget).
    Shows how rapidly each method captures transcription mistakes as reviewer
    effort increases.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=150)

    # Map colors and styles for each method
    styles = {
        "Random": {"color": "#64748B", "linestyle": ":", "marker": "s", "label": "Random Review (Diagonal Expectation)"},
        "Native": {"color": "#D97706", "linestyle": "--", "marker": "^", "label": "Whisper Native Confidence (avg_logprob)"},
        "Hybrid": {"color": "#2563EB", "linestyle": "-", "marker": "o", "label": "WhisperWatch Hybrid (50/50 Blend)"},
        "WhisperWatch": {"color": "#7C3AED", "linestyle": "-.", "marker": "d", "label": "WhisperWatch Classifier (risk_score)"},
    }

    unique_methods = curve_df["method"].unique()
    for method in unique_methods:
        sub = curve_df[curve_df["method"] == method].sort_values("budget")
        style = styles["WhisperWatch"]
        for key, s in styles.items():
            if key.lower() in method.lower() and (key != "Native" or "watch" not in method.lower()):
                style = s
                break

        x_vals = sub["budget"] * 100.0
        y_vals = sub["recall_at_budget"] * 100.0
        ax.plot(
            x_vals, y_vals,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
            label=style["label"],
        )

    # Highlight 20% budget reference line
    target_pct = target_budget * 100.0
    ax.axvline(
        x=target_pct,
        color="#DC2626",
        linestyle="-.",
        linewidth=1.4,
        alpha=0.7,
        label=f"Operational Review Budget ({int(target_pct)}%)",
    )

    ax.set_xlabel("Review Budget (% of Total Clips Inspected)", weight="bold", labelpad=8)
    ax.set_ylabel("High-Error Recall (% of All Severe Errors Caught)", weight="bold", labelpad=8)
    ax.set_title(title, pad=15, weight="bold")
    ax.set_xlim(0, max(curve_df["budget"].max() * 100.0 + 5, 55))
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_cross_dataset_comparison(
    summary_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "High-Error Recall @ 20% Budget Across Evaluation Cohorts",
) -> Path:
    """Generate a multi-cohort comparison chart comparing Native Confidence vs Hybrid.

    Shows Recall (%) for:
    - Native Whisper Confidence
    - WhisperWatch Hybrid (50/50 Percentile Blend)
    across all four cohorts with transparent cohort sample sizes and error counts.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)

    cohorts = summary_df["cohort"].tolist()
    native_rec = [r * 100.0 for r in summary_df["native_recall"]]
    hybrid_rec = [r * 100.0 for r in summary_df["hybrid_recall"]]

    x = np.arange(len(cohorts))
    width = 0.35

    rects1 = ax.bar(x - width / 2, native_rec, width, label="Whisper Native Confidence", color="#D97706")
    rects2 = ax.bar(x + width / 2, hybrid_rec, width, label="WhisperWatch Hybrid (50/50)", color="#2563EB")

    def autolabel(rects, counts):
        for rect, count in zip(rects, counts):
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%\n({count} caught)",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, weight="bold",
            )

    autolabel(rects1, summary_df["native_caught"].tolist())
    autolabel(rects2, summary_df["hybrid_caught"].tolist())

    ax.set_ylabel("High-Error Recall (% Caught at 20% Budget)", weight="bold")
    ax.set_title(title, pad=15, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, weight="medium", fontsize=9.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(max(native_rec + hybrid_rec) * 1.30, 45))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
