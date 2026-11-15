"""Frozen inference-only ranking math shared by the backend and development run."""

from math import ceil

import numpy as np
import pandas as pd


REVIEW_FRACTION = .20


def risk_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Higher scores mean higher priority. Missing confidence gets highest priority."""
    return pd.DataFrame({"boosted_trees": frame["risk_score"],
                         "whisper_confidence": -frame["avg_logprob"].fillna(-np.inf)})


def blend(signals: pd.DataFrame, boosted_weight: float) -> pd.Series:
    """Average percentile ranks using mean ranks for ties."""
    ranks = signals.rank(method="average", pct=True)
    return boosted_weight * ranks["boosted_trees"] + (1 - boosted_weight) * ranks["whisper_confidence"]


def ranked_ids(risk: pd.Series) -> pd.Index:
    return risk.sort_index().sort_values(ascending=False, kind="stable").index


def review_queue(risk: pd.Series) -> pd.Index:
    return ranked_ids(risk)[:ceil(len(risk) * REVIEW_FRACTION)]
