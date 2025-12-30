from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


@dataclass(frozen=True)
class CurveData:
    x: np.ndarray
    y: np.ndarray
    thresholds: np.ndarray


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # roc_auc_score fails if only one class exists.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> CurveData:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return CurveData(x=fpr, y=tpr, thresholds=thr)


def pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> CurveData:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    return CurveData(x=rec, y=prec, thresholds=thr)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

