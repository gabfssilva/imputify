"""Reconstruction metrics for imputation evaluation.

All metrics operate on masked positions only — y_true and y_pred
should contain values at positions that were originally missing.
"""

from typing import Literal, TypeAlias

import numpy as np

FeatureType: TypeAlias = Literal['categorical', 'numerical']

def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Error for numerical features.

    Args:
        y_true: Ground truth values at masked positions.
        y_pred: Imputed values at masked positions.

    Returns:
        Mean of absolute differences.
    """
    if len(y_true) == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root Mean Squared Error for numerical features.

    Args:
        y_true: Ground truth values at masked positions.
        y_pred: Imputed values at masked positions.

    Returns:
        Square root of mean squared differences.
    """
    if len(y_true) == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Normalized RMSE for numerical features.

    RMSE divided by the range of y_true. Returns 0.0 if the
    range is zero (constant column).

    Args:
        y_true: Ground truth values at masked positions.
        y_pred: Imputed values at masked positions.

    Returns:
        RMSE normalized to [0, 1] by ground truth range.
    """
    if len(y_true) == 0:
        return float('nan')
    range_val = np.max(y_true) - np.min(y_true)
    if range_val == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / range_val)


def categorical_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Exact match accuracy for categorical features.

    Args:
        y_true: Ground truth categories at masked positions.
        y_pred: Imputed categories at masked positions.

    Returns:
        Proportion of exact matches (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return float('nan')
    return float(np.mean(y_true == y_pred))
