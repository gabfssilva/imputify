"""Predictive metrics for imputation evaluation.

These metrics train machine learning models to assess whether imputed data
preserves predictive relationships. The philosophy: good imputation should
maintain the ability to predict target variables from features.
"""
from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

TaskType: TypeAlias = Literal['classification', 'regression']


def predictive_comparison(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    y: np.ndarray,
    X_missing: np.ndarray | None = None,
    estimator: BaseEstimator | None = None,
    task_type: TaskType | None = None,
    n_folds: int = 10,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare predictive performance on true vs imputed data using K-Fold CV.

    Trains models on original data (X), imputed data (X_imputed), and optionally
    on data with missing values (X_missing). Uses K-Fold cross-validation to
    provide robust estimates with uncertainty quantification.

    Args:
        X_true: Original feature matrix (n_samples, n_features)
        X_imputed: Imputed feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        X_missing: Data with missing values before imputation (n_samples, n_features).
                   If provided, trains a third model on data with NaN values.
                   Useful for estimators that handle missing values natively.
        estimator: sklearn estimator (if None, uses HistGradientBoosting which
                   handles NaN natively)
        task_type: 'classification' or 'regression' (auto-inferred if None)
        n_folds: Number of cross-validation folds (default: 10)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with model performance metrics (mean and std across folds):
        {
            'X': {'metric1': {'mean': float, 'std': float}, ...},
            'X_imputed': {'metric1': {'mean': float, 'std': float}, ...},
            'X_missing': {'metric1': {'mean': float, 'std': float}, ...}
        }

        Returns None if no features are available (X_true.shape[1] == 0).

    Example:
        >>> results = predictive_comparison(X_true, X_imputed, y, X_missing)
        >>> print(f"X R²: {results['X']['r2']['mean']:.3f} ± {results['X']['r2']['std']:.3f}")
        >>> print(f"X_imputed R²: {results['X_imputed']['r2']['mean']:.3f}")
    """
    if X_true.shape[1] == 0:
        return None

    if task_type is None:
        task_type = infer_task_type(y)

    if estimator is None:
        if task_type == 'classification':
            estimator = HistGradientBoostingClassifier(random_state=seed)
        else:
            estimator = HistGradientBoostingRegressor(random_state=seed)

    if task_type == 'classification':
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics_X = []
    fold_metrics_X_imputed = []
    fold_metrics_X_missing = []

    for train_idx, test_idx in kfold.split(X_true, y):
        y_train, y_test = y[train_idx], y[test_idx]

        model_true = clone(estimator)
        model_true.fit(X_true[train_idx], y_train)
        y_pred_true = model_true.predict(X_true[test_idx])

        model_imputed = clone(estimator)
        model_imputed.fit(X_imputed[train_idx], y_train)
        y_pred_imputed = model_imputed.predict(X_imputed[test_idx])

        if task_type == 'classification':
            fold_metrics_X.append(compute_classification_metrics(y_test, y_pred_true))
            fold_metrics_X_imputed.append(compute_classification_metrics(y_test, y_pred_imputed))
        else:
            fold_metrics_X.append(compute_regression_metrics(y_test, y_pred_true))
            fold_metrics_X_imputed.append(compute_regression_metrics(y_test, y_pred_imputed))

        if X_missing is not None:
            model_missing = clone(estimator)
            model_missing.fit(X_missing[train_idx], y_train)
            y_pred_missing = model_missing.predict(X_missing[test_idx])

            if task_type == 'classification':
                fold_metrics_X_missing.append(
                    compute_classification_metrics(y_test, y_pred_missing)
                )
            else:
                fold_metrics_X_missing.append(
                    compute_regression_metrics(y_test, y_pred_missing)
                )

    results = {
        'X': _aggregate_fold_metrics(fold_metrics_X),
        'X_imputed': _aggregate_fold_metrics(fold_metrics_X_imputed),
    }

    if fold_metrics_X_missing:
        results['X_missing'] = _aggregate_fold_metrics(fold_metrics_X_missing)

    return results


def _aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics from multiple folds into mean and std.

    Args:
        fold_metrics: List of metric dictionaries, one per fold

    Returns:
        Dictionary with mean and std for each metric:
        {'metric_name': {'mean': float, 'std': float}, ...}
    """
    if not fold_metrics:
        return {}

    metric_names = fold_metrics[0].keys()
    aggregated = {}

    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    return aggregated


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1
    """
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(
            precision_score(y_true, y_pred, average='weighted', zero_division=0)
        ),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics: r2, mae, mse, rmse
    """
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def infer_task_type(y: np.ndarray, classification_threshold: int = 20) -> TaskType:
    """Infer whether target represents classification or regression.

    Args:
        y: Target variable
        classification_threshold: Max unique values to consider classification.

    Returns:
        'classification' or 'regression'

    Logic:
        - String/object dtype → classification
        - ≤threshold unique values → classification
        - Otherwise → regression
    """
    if y.dtype == 'object' or y.dtype.kind in ['U', 'S']:
        return 'classification'

    n_unique = len(np.unique(y))
    if n_unique <= classification_threshold:
        return 'classification'

    return 'regression'



def predictive_r2_delta(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    seed: int | None = None,
) -> float:
    """Quick R² comparison for regression tasks.

    Returns:
        R² difference (X - X_imputed) mean across folds.
        Positive = imputation hurt performance.
    """
    results = predictive_comparison(
        X_true,
        X_imputed,
        y,
        task_type='regression',
        n_folds=n_folds,
        seed=seed,
    )
    return results['X']['r2']['mean'] - results['X_imputed']['r2']['mean']


def predictive_accuracy_delta(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    seed: int | None = None,
) -> float:
    """Quick accuracy comparison for classification tasks.

    Returns:
        Accuracy difference (X - X_imputed) mean across folds.
        Positive = imputation hurt performance.
    """
    results = predictive_comparison(
        X_true,
        X_imputed,
        y,
        task_type='classification',
        n_folds=n_folds,
        seed=seed,
    )
    return results['X']['accuracy']['mean'] - results['X_imputed']['accuracy']['mean']
