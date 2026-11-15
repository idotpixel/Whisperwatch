"""WhisperWatch Evaluation and Baseline Comparison Module.

This module provides reusable, standalone evaluation tools to answer the
WhisperWatch central research question:
"Can WhisperWatch identify more high-error transcriptions than a simple
Whisper-confidence ranking when human reviewers can inspect only a limited
number of clips?"

Key capabilities:
- Core classification metrics: Precision, Recall, F1, Confusion Matrix, PR-AUC.
- Class prevalence and imbalance assessment.
- Priority-based review-budget simulation (e.g. top 20% review capacity).
- Multi-baseline comparison:
    1. Random Review (expected fraction & empirical simulation).
    2. Native Whisper Confidence (ranked by avg_logprob ascending: lower logprob = higher priority).
    3. WhisperWatch ML Classifier (ranked by risk_score descending: higher score = higher priority).
- Review-budget sweep curves across variable budgets (e.g., 5% to 50%).
- Data validation and loading for predictions exported by Member A.
- Error analysis extraction (identifying where WhisperWatch succeeds or fails).
- Frontend data exporter for Member B's Gradio interface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Standard default threshold for high-error transcription (Word Error Rate > 20%)
DEFAULT_WER_THRESHOLD = 0.20
DEFAULT_REVIEW_BUDGET = 0.20


# ---------------------------------------------------------------------------
# 1. Core Classification Metrics
# ---------------------------------------------------------------------------

def compute_prevalence(y_true: Union[Sequence[int], np.ndarray, pd.Series]) -> float:
    """Compute prevalence (proportion of positive / high-error instances).

    Args:
        y_true: Ground truth binary labels (1 = high error, 0 = acceptable).

    Returns:
        Float prevalence between 0.0 and 1.0. Returns 0.0 if empty.
    """
    arr = np.asarray(y_true)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr == 1))


def compute_confusion_matrix(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> Dict[str, int]:
    """Compute confusion matrix components (TP, FP, TN, FN).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dictionary with keys 'tp', 'fp', 'tn', 'fn', 'total'.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if y_t.size == 0 or y_p.size == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}

    if y_t.shape != y_p.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}"
        )

    # Scikit-learn confusion_matrix returns [[tn, fp], [fn, tp]] for labels=[0, 1]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total": int(len(y_t)),
    }


def compute_precision(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """Compute precision: TP / (TP + FP)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if y_t.size == 0:
        return 0.0
    return float(precision_score(y_t, y_p, zero_division=zero_division))


def compute_recall(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """Compute recall (sensitivity): TP / (TP + FN)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if y_t.size == 0:
        return 0.0
    return float(recall_score(y_t, y_p, zero_division=zero_division))


def compute_f1(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """Compute F1 score: 2 * (P * R) / (P + R)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if y_t.size == 0:
        return 0.0
    return float(f1_score(y_t, y_p, zero_division=zero_division))


def compute_average_precision(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    scores: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """Compute Average Precision (PR-AUC).

    Handles edge cases (e.g., no positive labels) by returning 0.0.
    """
    y_t = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if y_t.size == 0 or s.size == 0:
        return 0.0
    if len(np.unique(y_t)) < 2:
        # If all 0s or all 1s, PR-AUC is undefined or trivial
        return float(np.mean(y_t))

    return float(average_precision_score(y_t, s))


def compute_roc_auc(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    scores: Union[Sequence[float], np.ndarray, pd.Series],
) -> Optional[float]:
    """Compute ROC-AUC score.

    Returns None if fewer than two classes are present.
    """
    y_t = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if y_t.size == 0 or s.size == 0 or len(np.unique(y_t)) < 2:
        return None

    return float(roc_auc_score(y_t, s))


def evaluate_classification_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    scores: Optional[Union[Sequence[float], np.ndarray, pd.Series]] = None,
    zero_division: float = 0.0,
) -> Dict[str, Any]:
    """Calculate a complete dictionary of standard classification metrics."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    cm = compute_confusion_matrix(y_t, y_p)
    prec = compute_precision(y_t, y_p, zero_division=zero_division)
    rec = compute_recall(y_t, y_p, zero_division=zero_division)
    f1 = compute_f1(y_t, y_p, zero_division=zero_division)
    prev = compute_prevalence(y_t)

    result: Dict[str, Any] = {
        "prevalence": prev,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tp": cm["tp"],
        "fp": cm["fp"],
        "tn": cm["tn"],
        "fn": cm["fn"],
        "total_samples": cm["total"],
    }

    if scores is not None:
        result["pr_auc"] = compute_average_precision(y_t, scores)
        result["roc_auc"] = compute_roc_auc(y_t, scores)

    return result


# ---------------------------------------------------------------------------
# 2. Hybrid Risk Score and Paired Comparisons
# ---------------------------------------------------------------------------

def compute_hybrid_risk(
    df: pd.DataFrame,
    classifier_risk_col: str = "classifier_risk",
    avg_logprob_col: str = "avg_logprob",
    missing_confidence_val: float = float("-inf"),
) -> pd.Series:
    """Compute 50/50 percentile hybrid risk score.

    Combines:
    1. Native risk: -avg_logprob with missing confidence ranked first (highest priority).
    2. Classifier risk: classifier_risk ranked ascending.
    Percentile ranks computed using average ties (pct=True).
    Hybrid risk is the unweighted mean of the two percentiles.

    Args:
        df: Input DataFrame.
        classifier_risk_col: Column name for classifier risk score.
        avg_logprob_col: Column name for Whisper avg_logprob.
        missing_confidence_val: Fill value for avg_logprob before negation.
            Default float("-inf") negates to +inf, giving missing confidence rank 1.0.

    Returns:
        pd.Series containing computed hybrid risk scores in range [0.0, 1.0].
    """
    if df.empty:
        return pd.Series(dtype=float)

    if classifier_risk_col not in df.columns:
        raise KeyError(f"Classifier risk column '{classifier_risk_col}' not found in DataFrame.")
    if avg_logprob_col not in df.columns:
        raise KeyError(f"Whisper logprob column '{avg_logprob_col}' not found in DataFrame.")

    native = -df[avg_logprob_col].fillna(missing_confidence_val)
    pct_native = native.rank(method="average", pct=True)
    pct_classifier = df[classifier_risk_col].rank(method="average", pct=True)

    return (pct_native + pct_classifier) / 2.0


def compute_paired_comparison(
    df: pd.DataFrame,
    method_a_review_col: str,
    method_b_review_col: str,
    true_col: str = "true_high_error",
) -> Dict[str, Any]:
    """Compute discordant pair analysis and exact binomial test for two review selections.

    Compares two methods on the identical population of high-error transcripts:
    - n_both (both caught)
    - n_a_only (Method A caught, Method B missed)
    - n_b_only (Method B caught, Method A missed)
    - n_neither (neither caught)
    - p_value: Two-sided exact binomial test on discordant pairs.
    """
    if df.empty or true_col not in df.columns:
        return {
            "n_high_error": 0,
            "both_caught": 0,
            "a_only_caught": 0,
            "b_only_caught": 0,
            "neither_caught": 0,
            "net_difference": 0,
            "p_value": 1.0,
        }

    high_err = df[df[true_col] == 1]
    n_he = len(high_err)
    if n_he == 0:
        return {
            "n_high_error": 0,
            "both_caught": 0,
            "a_only_caught": 0,
            "b_only_caught": 0,
            "neither_caught": 0,
            "net_difference": 0,
            "p_value": 1.0,
        }

    a_flag = high_err[method_a_review_col].astype(bool)
    b_flag = high_err[method_b_review_col].astype(bool)

    both = int((a_flag & b_flag).sum())
    a_only = int((a_flag & (~b_flag)).sum())
    b_only = int(((~a_flag) & b_flag).sum())
    neither = int(((~a_flag) & (~b_flag)).sum())

    total_discordant = a_only + b_only
    if total_discordant > 0:
        res = binomtest(b_only, total_discordant, 0.5)
        p_val = float(res.pvalue)
    else:
        p_val = 1.0

    return {
        "n_high_error": n_he,
        "both_caught": both,
        "a_only_caught": a_only,
        "b_only_caught": b_only,
        "neither_caught": neither,
        "net_difference": b_only - a_only,
        "p_value": p_val,
    }


# ---------------------------------------------------------------------------
# 3. Review-Budget Prioritization
# ---------------------------------------------------------------------------

def evaluate_review_budget(
    df: pd.DataFrame,
    priority_col: str,
    budget: float = DEFAULT_REVIEW_BUDGET,
    ascending: bool = False,
    true_col: str = "true_high_error",
    wer_col: Optional[str] = "wer",
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
    rounding: str = "round",
    budget_count: Optional[int] = None,
    tie_break_col: Optional[str] = "clip_id",
    na_position: str = "last",
) -> Dict[str, Any]:
    """Simulate a review-budget scenario for a given prioritization column.

    A reviewer has capacity to inspect a fixed fraction `budget` (e.g. 20%)
    of clips, or an explicit count `budget_count`. Clips are ranked by `priority_col`.

    Direction handling:
    - For `risk_score` (higher = higher error risk): ascending=False
    - For `avg_logprob` (lower = lower confidence = higher error risk): ascending=True
    - Missing confidence handling: if na_position='first', missing values are placed
      first in the priority queue.

    Args:
        df: DataFrame containing the records.
        priority_col: Name of column to sort by.
        budget: Fraction of total clips to review (0.0 to 1.0).
        ascending: True if smaller values indicate higher priority (e.g. avg_logprob).
                   False if larger values indicate higher priority (e.g. risk_score).
        true_col: Column containing ground truth high-error binary flag (0 or 1).
        wer_col: Optional column containing raw WER to derive true_col if needed.
        wer_threshold: Threshold to define high error (WER > wer_threshold).
        rounding: Rounding method ('round', 'ceil', 'floor').
        budget_count: Explicit number of clips to review (overrides budget fraction).
        tie_break_col: Secondary sort column for deterministic tie-breaking.
        na_position: 'last' or 'first' for NaN scores.

    Returns:
        Dictionary with review budget evaluation metrics:
        - budget_fraction: Requested fraction.
        - total_clips: Total number of clips.
        - reviewed_clips: Number of clips selected for review.
        - total_high_error: Total actual high-error clips in the dataset.
        - high_error_caught: High-error clips caught in the reviewed budget.
        - high_error_missed: High-error clips that escaped human review.
        - recall_at_budget: Fraction of all high-error clips caught (caught / total_high_error).
        - precision_at_budget: Fraction of reviewed clips that were genuinely high-error.
        - lift_over_random: recall_at_budget / budget_fraction.
    """
    if df.empty:
        return {
            "budget_fraction": budget,
            "total_clips": 0,
            "reviewed_clips": 0,
            "total_high_error": 0,
            "high_error_caught": 0,
            "high_error_missed": 0,
            "recall_at_budget": 0.0,
            "precision_at_budget": 0.0,
            "lift_over_random": 1.0,
        }

    work_df = df.copy()

    # Ensure ground truth binary label is available
    if true_col not in work_df.columns:
        if wer_col in work_df.columns:
            work_df[true_col] = (work_df[wer_col] > wer_threshold).astype(int)
        else:
            raise KeyError(
                f"Neither ground truth column '{true_col}' nor WER column '{wer_col}' "
                "found in DataFrame."
            )

    # Check priority column exists
    if priority_col not in work_df.columns:
        raise KeyError(f"Priority column '{priority_col}' not found in DataFrame.")

    # Drop NaNs based on na_position
    if na_position == "first":
        valid_mask = work_df[true_col].notna()
    else:
        valid_mask = work_df[priority_col].notna() & work_df[true_col].notna()
    work_df = work_df[valid_mask]

    n_total = len(work_df)
    if n_total == 0:
        return {
            "budget_fraction": budget,
            "total_clips": 0,
            "reviewed_clips": 0,
            "total_high_error": 0,
            "high_error_caught": 0,
            "high_error_missed": 0,
            "recall_at_budget": 0.0,
            "precision_at_budget": 0.0,
            "lift_over_random": 1.0,
        }

    # Determine number of clips to review
    if budget_count is not None:
        k = max(0, min(n_total, int(round(budget_count))))
    elif rounding == "ceil":
        k = int(math.ceil(n_total * budget))
    elif rounding == "floor":
        k = int(math.floor(n_total * budget))
    else:
        k = int(round(n_total * budget))
    k = max(0, min(n_total, k))

    total_high_error = int(work_df[true_col].sum())

    # Deterministic sorting
    sort_cols = [priority_col]
    asc_list = [ascending]
    if tie_break_col and tie_break_col in work_df.columns and tie_break_col != priority_col:
        sort_cols.append(tie_break_col)
        asc_list.append(True)

    sorted_df = work_df.sort_values(by=sort_cols, ascending=asc_list, na_position=na_position, kind="mergesort")

    # Select top K
    reviewed = sorted_df.iloc[:k]
    high_error_caught = int(reviewed[true_col].sum()) if k > 0 else 0
    high_error_missed = total_high_error - high_error_caught

    recall_at_budget = (
        (high_error_caught / total_high_error) if total_high_error > 0 else 0.0
    )
    precision_at_budget = (high_error_caught / k) if k > 0 else 0.0
    effective_budget = (k / n_total) if n_total > 0 else budget
    lift_over_random = (
        (recall_at_budget / effective_budget) if (effective_budget > 0 and total_high_error > 0) else 1.0
    )

    return {
        "budget_fraction": budget,
        "total_clips": n_total,
        "reviewed_clips": k,
        "total_high_error": total_high_error,
        "high_error_caught": high_error_caught,
        "high_error_missed": high_error_missed,
        "recall_at_budget": float(recall_at_budget),
        "precision_at_budget": float(precision_at_budget),
        "lift_over_random": float(lift_over_random),
    }


# ---------------------------------------------------------------------------
# 4. Multi-Baseline Comparison (Random vs Whisper Native vs WhisperWatch)
# ---------------------------------------------------------------------------

def compare_baselines(
    df: pd.DataFrame,
    budget: float = DEFAULT_REVIEW_BUDGET,
    risk_col: str = "risk_score",
    logprob_col: str = "avg_logprob",
    true_col: str = "true_high_error",
    wer_col: Optional[str] = "wer",
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
    random_seed: int = 42,
    n_random_simulations: int = 100,
    rounding: str = "round",
    budget_count: Optional[int] = None,
    tie_break_col: Optional[str] = "clip_id",
    hybrid_col: Optional[str] = "original_hybrid_risk",
    include_hybrid: Optional[bool] = None,
    na_position_logprob: str = "first",
) -> pd.DataFrame:
    """Compare methods on the exact same held-out records.

    Methods compared:
    1. Random Review: Expected recall is equal to budget fraction (e.g. 20%).
       Precision is equal to the prevalence of high-error transcripts in the dataset.
    2. Native Whisper Confidence: Ranked by avg_logprob ascending (lower = higher priority).
    3. WhisperWatch Model: Ranked by risk_score descending (higher = higher priority).
    4. WhisperWatch Original Hybrid (optional / auto-detected): 50/50 percentile rank blending.

    Returns:
        pd.DataFrame with comparative columns:
        - method: Method name
        - budget_fraction: Budget percentage (e.g. 0.20)
        - clips_reviewed: Number of clips inspected
        - total_high_error: Total high-error clips in the dataset
        - high_error_caught: Clips with WER > 20% caught
        - recall_at_budget: % of all high-error clips caught
        - precision_at_budget: % of reviewed clips that were genuinely high error
        - lift_over_random: Multiplier over random guessing
        - relative_improvement_vs_whisper: Percentage improvement in recall over Whisper baseline
    """
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()

    # Ensure ground truth binary label is computed
    if true_col not in work_df.columns:
        if wer_col in work_df.columns:
            work_df[true_col] = (work_df[wer_col] > wer_threshold).astype(int)
        else:
            raise KeyError(
                f"Neither ground truth column '{true_col}' nor WER column '{wer_col}' "
                "found in DataFrame."
            )

    # Resolve risk column if alternative name exists
    if risk_col not in work_df.columns:
        if "classifier_risk" in work_df.columns:
            risk_col = "classifier_risk"
        elif "original_hybrid_risk" in work_df.columns:
            risk_col = "original_hybrid_risk"
        else:
            raise KeyError(f"Risk column '{risk_col}' not found in DataFrame.")

    # Resolve logprob column if native_risk exists
    if logprob_col not in work_df.columns and "native_risk" in work_df.columns:
        work_df[logprob_col] = -work_df["native_risk"]

    if logprob_col not in work_df.columns:
        raise KeyError(f"Logprob column '{logprob_col}' not found in DataFrame.")

    # Population filtering: require valid true label and risk_col
    if na_position_logprob == "first":
        valid_mask = work_df[true_col].notna() & work_df[risk_col].notna()
    else:
        valid_mask = (
            work_df[true_col].notna()
            & work_df[risk_col].notna()
            & work_df[logprob_col].notna()
        )
    work_df = work_df[valid_mask].copy()

    n_total = len(work_df)
    total_high_error = int(work_df[true_col].sum())
    prevalence = total_high_error / n_total if n_total > 0 else 0.0

    if budget_count is not None:
        k = max(0, min(n_total, int(round(budget_count))))
    elif rounding == "ceil":
        k = int(math.ceil(n_total * budget))
    elif rounding == "floor":
        k = int(math.floor(n_total * budget))
    else:
        k = int(round(n_total * budget))
    k = max(0, min(n_total, k))

    # --- Method 1: Random Review ---
    expected_random_caught = total_high_error * (k / n_total) if n_total > 0 else 0.0
    expected_random_recall = (k / n_total) if (n_total > 0 and total_high_error > 0) else 0.0
    expected_random_precision = prevalence

    random_res = {
        "method": "Random Review (Baseline 1)",
        "priority_signal": "Uniform Random Selection",
        "budget_fraction": budget,
        "clips_reviewed": k,
        "total_high_error": total_high_error,
        "high_error_caught": round(expected_random_caught, 1),
        "recall_at_budget": float(expected_random_recall),
        "precision_at_budget": float(expected_random_precision),
        "lift_over_random": 1.0,
        "relative_improvement_vs_whisper": 0.0,
    }

    # --- Method 2: Native Whisper Confidence ---
    whisper_eval = evaluate_review_budget(
        work_df,
        priority_col=logprob_col,
        budget=budget,
        budget_count=k,
        ascending=True,  # Lower logprob = higher priority
        true_col=true_col,
        rounding=rounding,
        tie_break_col=tie_break_col,
        na_position=na_position_logprob,
    )
    whisper_res = {
        "method": "Whisper Native Confidence (Baseline 2)",
        "priority_signal": f"Ascending {logprob_col} (Least Confident First)",
        "budget_fraction": budget,
        "clips_reviewed": whisper_eval["reviewed_clips"],
        "total_high_error": total_high_error,
        "high_error_caught": whisper_eval["high_error_caught"],
        "recall_at_budget": whisper_eval["recall_at_budget"],
        "precision_at_budget": whisper_eval["precision_at_budget"],
        "lift_over_random": whisper_eval["lift_over_random"],
        "relative_improvement_vs_whisper": 0.0,
    }

    # --- Method 3: WhisperWatch ML Classifier ---
    ww_eval = evaluate_review_budget(
        work_df,
        priority_col=risk_col,
        budget=budget,
        budget_count=k,
        ascending=False,  # Higher risk score = higher priority
        true_col=true_col,
        rounding=rounding,
        tie_break_col=tie_break_col,
    )

    whisper_caught = whisper_eval["high_error_caught"]
    ww_caught = ww_eval["high_error_caught"]
    if whisper_caught > 0:
        rel_improvement = ((ww_caught - whisper_caught) / whisper_caught) * 100.0
    else:
        rel_improvement = 0.0 if ww_caught == 0 else 100.0

    ww_res = {
        "method": "WhisperWatch (Our Model)",
        "priority_signal": f"Descending {risk_col} (Highest Risk First)",
        "budget_fraction": budget,
        "clips_reviewed": ww_eval["reviewed_clips"],
        "total_high_error": total_high_error,
        "high_error_caught": ww_eval["high_error_caught"],
        "recall_at_budget": ww_eval["recall_at_budget"],
        "precision_at_budget": ww_eval["precision_at_budget"],
        "lift_over_random": ww_eval["lift_over_random"],
        "relative_improvement_vs_whisper": rel_improvement,
    }

    method_rows = [random_res, whisper_res, ww_res]

    # --- Method 4: Optional Original Hybrid ---
    should_include_hybrid = (
        include_hybrid is True
        or (include_hybrid is None and hybrid_col is not None and hybrid_col in work_df.columns)
    )
    if should_include_hybrid:
        if hybrid_col not in work_df.columns:
            work_df[hybrid_col] = compute_hybrid_risk(work_df, classifier_risk_col=risk_col, avg_logprob_col=logprob_col)
        hyb_eval = evaluate_review_budget(
            work_df,
            priority_col=hybrid_col,
            budget=budget,
            budget_count=k,
            ascending=False,
            true_col=true_col,
            rounding=rounding,
            tie_break_col=tie_break_col,
        )
        hyb_caught = hyb_eval["high_error_caught"]
        hyb_rel = ((hyb_caught - whisper_caught) / whisper_caught * 100.0) if whisper_caught > 0 else 0.0
        hyb_res = {
            "method": "WhisperWatch Original Hybrid",
            "priority_signal": f"Descending {hybrid_col} (50/50 Percentile Blend)",
            "budget_fraction": budget,
            "clips_reviewed": hyb_eval["reviewed_clips"],
            "total_high_error": total_high_error,
            "high_error_caught": hyb_eval["high_error_caught"],
            "recall_at_budget": hyb_eval["recall_at_budget"],
            "precision_at_budget": hyb_eval["precision_at_budget"],
            "lift_over_random": hyb_eval["lift_over_random"],
            "relative_improvement_vs_whisper": hyb_rel,
        }
        method_rows.append(hyb_res)

    results = pd.DataFrame(method_rows)
    return results


# ---------------------------------------------------------------------------
# 4. Review-Budget Effectiveness Sweep Curves
# ---------------------------------------------------------------------------

def evaluate_budget_curve(
    df: pd.DataFrame,
    budgets: Optional[Sequence[float]] = None,
    risk_col: str = "risk_score",
    logprob_col: str = "avg_logprob",
    true_col: str = "true_high_error",
    wer_col: Optional[str] = "wer",
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
) -> pd.DataFrame:
    """Evaluate performance across a sweep of review budgets (e.g. 5% to 50%).

    Useful for plotting the review-budget effectiveness curve showing how
    many high-error clips are captured as reviewer capacity expands.
    """
    if budgets is None:
        budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows = []
    for b in budgets:
        comp = compare_baselines(
            df=df,
            budget=b,
            risk_col=risk_col,
            logprob_col=logprob_col,
            true_col=true_col,
            wer_col=wer_col,
            wer_threshold=wer_threshold,
        )
        if not comp.empty:
            for _, r in comp.iterrows():
                rows.append({
                    "budget": b,
                    "method": r["method"],
                    "clips_reviewed": r["clips_reviewed"],
                    "high_error_caught": r["high_error_caught"],
                    "recall_at_budget": r["recall_at_budget"],
                    "precision_at_budget": r["precision_at_budget"],
                    "lift_over_random": r["lift_over_random"],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Data Integrity and Prediction File Loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["risk_score", "avg_logprob"]
RECOMMENDED_COLUMNS = [
    "clip_id",
    "language",
    "true_high_error",
    "risk_score",
    "avg_logprob",
    "wer",
    "whisper_text",
    "human_text",
]


def load_predictions(
    file_path: Union[str, Path],
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
    filter_language: Optional[str] = "en",
    drop_na: bool = True,
    check_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate model prediction files exported by Member A.

    Supports CSV, JSON, and Parquet file formats.
    Validates column integrity, checks for duplicate clips, calculates
    ground-truth error flags from WER if missing, and optionally filters by language.

    Args:
        file_path: Path to prediction file.
        wer_threshold: Threshold to define high error (WER > wer_threshold).
        filter_language: If provided (e.g. 'en'), filters dataset to this language.
                         Pass None to keep all languages.
        drop_na: If True, drops rows missing critical priority signals or ground truth.
        check_duplicates: If True, warns and deduplicates on 'clip_id'.

    Returns:
        Tuple of:
        - Cleaned pd.DataFrame ready for evaluation.
        - Metadata dictionary detailing data integrity statistics.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    # Load file based on extension
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".json", ".jsonl"]:
        lines = path.suffix.lower() == ".jsonl"
        df = pd.read_json(path, lines=lines)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Default try CSV
        df = pd.read_csv(path)

    initial_count = len(df)
    metadata: Dict[str, Any] = {
        "file_path": str(path),
        "initial_records": initial_count,
        "language_filtered": filter_language,
        "wer_threshold": wer_threshold,
    }

    # Map synonyms if standard names not directly present
    if "risk_score" not in df.columns:
        if "classifier_risk" in df.columns:
            df["risk_score"] = df["classifier_risk"]
        elif "original_hybrid_risk" in df.columns:
            df["risk_score"] = df["original_hybrid_risk"]

    if "avg_logprob" not in df.columns and "native_risk" in df.columns:
        df["avg_logprob"] = -df["native_risk"]

    if "whisper_text" not in df.columns and "transcript" in df.columns:
        df["whisper_text"] = df["transcript"]

    if "human_text" not in df.columns and "reference" in df.columns:
        df["human_text"] = df["reference"]

    # Validate essential columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"Missing required prediction column '{col}'. Available columns: {list(df.columns)}"
            )

    # Check for ground truth
    if "true_high_error" not in df.columns:
        if "wer" in df.columns:
            df["true_high_error"] = (df["wer"] > wer_threshold).astype(int)
        else:
            raise KeyError(
                "Neither 'true_high_error' nor 'wer' column found in predictions."
            )
    else:
        # Ensure it is standard 0 / 1 int
        df["true_high_error"] = df["true_high_error"].astype(int)

    # Filter language if requested and column exists
    if filter_language is not None and "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == filter_language.lower()].copy()
        metadata["records_after_language_filter"] = len(df)
    else:
        metadata["records_after_language_filter"] = len(df)

    # Deduplicate on clip_id if present
    if check_duplicates and "clip_id" in df.columns:
        duplicates_count = int(df["clip_id"].duplicated().sum())
        metadata["duplicate_clip_ids_removed"] = duplicates_count
        if duplicates_count > 0:
            df = df.drop_duplicates(subset=["clip_id"], keep="first").copy()
    else:
        metadata["duplicate_clip_ids_removed"] = 0

    # Handle missing values
    if drop_na:
        if "native_confidence_missing" in df.columns:
            valid_rows = df["true_high_error"].notna() & df["risk_score"].notna() & (df["avg_logprob"].notna() | df["native_confidence_missing"])
        else:
            valid_rows = df["true_high_error"].notna() & df["risk_score"].notna() & df["avg_logprob"].notna()
        na_count = int((~valid_rows).sum())
        metadata["missing_value_rows_dropped"] = na_count
        df = df[valid_rows].copy()
    else:
        metadata["missing_value_rows_dropped"] = 0

    final_count = len(df)
    metadata["final_records"] = final_count
    metadata["total_high_error"] = int(df["true_high_error"].sum()) if final_count > 0 else 0
    metadata["prevalence"] = (
        float(metadata["total_high_error"] / final_count) if final_count > 0 else 0.0
    )

    return df, metadata


# ---------------------------------------------------------------------------
# 6. Pedagogical Error and Case Analysis
# ---------------------------------------------------------------------------

def extract_error_analysis_cases(
    df: pd.DataFrame,
    budget: float = DEFAULT_REVIEW_BUDGET,
    risk_col: str = "risk_score",
    logprob_col: str = "avg_logprob",
    true_col: str = "true_high_error",
    n_samples: int = 5,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Extract key pedagogical cases highlighting where models succeed and fail.

    Categories:
    1. 'whisperwatch_wins': Clips where Whisper was overconfident (high logprob),
       but transcript had severe errors (high WER), and WhisperWatch correctly flagged it.
    2. 'false_alarms_avoided': Clips where Whisper had low logprob (unconfident),
       but transcript was actually accurate (low WER), and WhisperWatch correctly did NOT flag it.
    3. 'whisperwatch_false_alarms': Accurate transcripts that WhisperWatch unnecessarily flagged.
    4. 'missed_errors': Severe errors that both Whisper and WhisperWatch failed to catch.
    """
    if df.empty:
        return {}

    work_df = df.copy()
    n_total = len(work_df)
    k = max(1, int(round(n_total * budget)))

    # Determine thresholds based on review budget
    sorted_ww = work_df.sort_values(by=risk_col, ascending=False, kind="mergesort")
    ww_threshold = sorted_ww.iloc[k - 1][risk_col]
    work_df["flagged_ww"] = (work_df[risk_col] >= ww_threshold).astype(int)

    sorted_whisper = work_df.sort_values(by=logprob_col, ascending=True, kind="mergesort")
    whisper_threshold = sorted_whisper.iloc[k - 1][logprob_col]
    work_df["flagged_whisper"] = (work_df[logprob_col] <= whisper_threshold).astype(int)

    is_high_error = work_df[true_col] == 1
    is_low_error = work_df[true_col] == 0

    # 1. WhisperWatch Wins (High error, WW flagged, Whisper missed)
    ww_wins = work_df[is_high_error & (work_df["flagged_ww"] == 1) & (work_df["flagged_whisper"] == 0)]

    # 2. False alarms avoided (Low error, Whisper flagged, WW did not flag)
    false_alarms_avoided = work_df[
        is_low_error & (work_df["flagged_whisper"] == 1) & (work_df["flagged_ww"] == 0)
    ]

    # 3. WhisperWatch False Alarms (Low error, WW flagged)
    ww_false_alarms = work_df[is_low_error & (work_df["flagged_ww"] == 1)]

    # 4. Missed errors (High error, neither flagged)
    missed_errors = work_df[
        is_high_error & (work_df["flagged_ww"] == 0) & (work_df["flagged_whisper"] == 0)
    ]

    def sample_df(d: pd.DataFrame) -> pd.DataFrame:
        if len(d) <= n_samples:
            return d
        return d.sample(n=n_samples, random_state=random_seed)

    return {
        "whisperwatch_wins": sample_df(ww_wins),
        "false_alarms_avoided": sample_df(false_alarms_avoided),
        "whisperwatch_false_alarms": sample_df(ww_false_alarms),
        "missed_errors": sample_df(missed_errors),
    }


# ---------------------------------------------------------------------------
# 7. Frontend Example Exporter (for Member B's Gradio Interface)
# ---------------------------------------------------------------------------

def export_frontend_examples(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    budget: float = DEFAULT_REVIEW_BUDGET,
    risk_col: str = "risk_score",
    true_col: str = "true_high_error",
    wer_col: str = "wer",
    n_examples: int = 50,
    random_seed: int = 42,
) -> Path:
    """Generate a clean JSON file containing curated examples for Member B's Gradio app.

    Columns exported for each example:
    - clip_id: Identifier
    - whisper_text: What Whisper transcribed
    - human_text: True reference transcript (for historical demo display)
    - risk_score: WhisperWatch predicted error-risk score (0.0 to 1.0)
    - wer: Ground truth Word Error Rate
    - true_high_error: Ground truth error indicator (1 = High Error, 0 = Acceptable)
    - recommend_review: Boolean recommendation based on budget threshold
    - outcome_category: 'True Positive', 'False Positive', 'True Negative', 'False Negative'

    Args:
        df: Predictions DataFrame.
        output_path: Path to save JSON.
        budget: Review budget used to establish decision threshold.
        risk_col: Risk score column name.
        true_col: True high error column name.
        wer_col: WER column name.
        n_examples: Maximum number of diverse examples to export.
        random_seed: RNG seed for reproducible sampling.

    Returns:
        Path to exported JSON file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()
    n_total = len(work_df)

    if n_total == 0:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        return out_file

    k = max(1, int(round(n_total * budget)))
    sorted_df = work_df.sort_values(by=risk_col, ascending=False, kind="mergesort")
    threshold = float(sorted_df.iloc[k - 1][risk_col])

    work_df["recommend_review"] = work_df[risk_col] >= threshold

    # Determine classification outcome
    def get_outcome(row: pd.Series) -> str:
        rec = bool(row["recommend_review"])
        truth = int(row[true_col]) == 1
        if rec and truth:
            return "True Positive (High Risk Correctly Flagged)"
        if rec and not truth:
            return "False Positive (Unnecessary Review)"
        if not rec and truth:
            return "False Negative (High Risk Missed)"
        return "True Negative (Acceptable Quality Verified)"

    work_df["outcome_category"] = work_df.apply(get_outcome, axis=1)

    # Balance sampling across outcome categories so the demo showcases all quadrants
    categories = work_df["outcome_category"].unique()
    samples_per_cat = max(1, n_examples // len(categories))
    sampled_dfs = []

    for cat in categories:
        sub = work_df[work_df["outcome_category"] == cat]
        n_s = min(len(sub), samples_per_cat)
        sampled_dfs.append(sub.sample(n=n_s, random_state=random_seed))

    demo_df = pd.concat(sampled_dfs).drop_duplicates()

    # If we need more samples to reach n_examples, sample remaining
    if len(demo_df) < n_examples and len(work_df) > len(demo_df):
        remaining = work_df.loc[~work_df.index.isin(demo_df.index)]
        extra = remaining.sample(
            n=min(len(remaining), n_examples - len(demo_df)),
            random_state=random_seed,
        )
        demo_df = pd.concat([demo_df, extra])

    # Select standard output columns
    export_cols = [
        col for col in [
            "clip_id",
            "whisper_text",
            "human_text",
            "risk_score",
            "avg_logprob",
            wer_col,
            true_col,
            "recommend_review",
            "outcome_category",
        ]
        if col in demo_df.columns
    ]

    records = demo_df[export_cols].to_dict(orient="records")

    # Serialize cleanly
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    return out_file
