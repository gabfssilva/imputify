"""Complete evaluation function aggregating all three perspectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .reconstruction import mae, rmse, nrmse, categorical_accuracy
from .distribution import (
    wasserstein_distance,
    ks_statistic,
    kl_divergence,
    correlation_shift,
)
from .predictive import predictive_comparison
from imputify.shared import global_seed


@dataclass
class EvaluationResults:
    """Container for complete evaluation results across all perspectives."""

    reconstruction: dict[str, Any]
    distribution: dict[str, Any]
    predictive: dict[str, Any] | None = None

    @property
    def overall_reconstruction_score(self) -> float:
        """Average of all reconstruction metric averages."""
        scores = []
        for metric_name, metric_data in self.reconstruction.items():
            if isinstance(metric_data, dict) and 'avg' in metric_data:
                scores.append(metric_data['avg'])
        return float(np.mean(scores)) if scores else 0.0

    @property
    def overall_distribution_score(self) -> float:
        """Average of all distribution metric averages."""
        scores = []
        for metric_name, metric_data in self.distribution.items():
            if isinstance(metric_data, dict) and 'avg' in metric_data:
                scores.append(metric_data['avg'])
            elif isinstance(metric_data, float):
                scores.append(metric_data)
        return float(np.mean(scores)) if scores else 0.0

    @property
    def overall_predictive_score(self) -> float | None:
        """Average accuracy/r2 delta from predictive comparison (uses mean across folds)."""
        if self.predictive is None:
            return None

        deltas = [
            self.predictive['X'][key]['mean'] - self.predictive['X_imputed'][key]['mean']
            for key in self.predictive['X'].keys()
        ]
        # Lower delta is better (imputation had less impact)
        # Normalize: 1 - delta (if delta is small, score close to 1)
        normalized = [1.0 - min(abs(d), 1.0) for d in deltas]
        return float(np.mean(normalized)) if normalized else None

    @property
    def overall_predictive_std(self) -> float | None:
        """Average std across all predictive metrics (uncertainty estimate)."""
        if self.predictive is None:
            return None

        stds = [
            self.predictive['X_imputed'][key]['std']
            for key in self.predictive['X_imputed'].keys()
        ]
        return float(np.mean(stds)) if stds else None

    @property
    def overall_score(self) -> float:
        """Aggregate normalized score across all available perspectives (higher is better)."""
        recon_normalized = 1.0 / (1.0 + self.overall_reconstruction_score)
        dist_normalized = 1.0 / (1.0 + self.overall_distribution_score)

        scores = [recon_normalized, dist_normalized]

        if self.overall_predictive_score is not None:
            scores.append(self.overall_predictive_score)

        return float(np.mean(scores))

    def to_dict(self) -> dict[str, Any]:
        """Flatten results to dictionary."""
        result = {
            'reconstruction': self.reconstruction,
            'distribution': self.distribution,
            'summaries': {
                'overall_reconstruction': self.overall_reconstruction_score,
                'overall_distribution': self.overall_distribution_score,
                'overall': self.overall_score,
            },
        }

        if self.predictive is not None:
            result['predictive'] = self.predictive
            result['summaries']['overall_predictive'] = self.overall_predictive_score
            result['summaries']['overall_predictive_std'] = self.overall_predictive_std

        return result


def evaluate(
    *,
    X: pd.DataFrame,
    X_imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    y: np.ndarray | None = None,
    X_missing: pd.DataFrame | None = None,
    estimator: BaseEstimator | None = None,
) -> EvaluationResults:
    """Complete evaluation across reconstruction, distribution, and predictive perspectives.

    Args:
        X: Original DataFrame with no missing values
        X_imputed: DataFrame with imputed values
        missing_mask: Boolean DataFrame indicating which positions had missing values
        y: Optional target variable for predictive evaluation
        X_missing: Optional DataFrame with missing values (before imputation).
                   If provided, also evaluates predictive performance on data with NaN.
        estimator: Optional sklearn estimator for predictive evaluation.
                   If None, uses HistGradientBoostingClassifier/Regressor (handles NaN natively).

    Returns:
        EvaluationResults with detailed breakdown and aggregate scores

    Predictive Metrics:
        - Uses numeric columns as-is
        - Automatically label-encodes categorical columns
        - Skipped if no features available (e.g., datetime-only data)

    Example:
        >>> results = evaluate(X, X_imputed, missing_mask, y, X_missing)
        >>> print(f"Overall: {results.overall_score:.3f}")
        >>> print(f"Reconstruction: {results.overall_reconstruction_score:.3f}")
        >>> print(f"MAE per column: {results.reconstruction['mae']}")
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    reconstruction_scores = _evaluate_reconstruction(
        X, X_imputed, missing_mask, numeric_cols, categorical_cols
    )

    distribution_scores = _evaluate_distribution(
        X, X_imputed, missing_mask, numeric_cols, categorical_cols
    )

    predictive_scores = None
    if y is not None:
        X_true_arr, X_imputed_arr, X_missing_arr = _prepare_predictive_features(
            X, X_imputed, numeric_cols, categorical_cols, X_missing
        )

        predictive_scores = predictive_comparison(
            X_true_arr,
            X_imputed_arr,
            y,
            X_missing=X_missing_arr,
            estimator=estimator,
            seed=global_seed(),
        )

    return EvaluationResults(
        reconstruction=reconstruction_scores,
        distribution=distribution_scores,
        predictive=predictive_scores,
    )


def _prepare_predictive_features(
    X: pd.DataFrame,
    X_imputed: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    X_missing: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Prepare features for predictive evaluation.

    Combines numeric columns as-is with ordinal-encoded categorical columns.
    Uses OrdinalEncoder with encoded_missing_value=np.nan to preserve NaN.

    Args:
        X: Original DataFrame
        X_imputed: Imputed DataFrame
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        X_missing: Optional DataFrame with missing values (before imputation)

    Returns:
        Tuple of (X_true, X_imputed, X_missing) as numpy arrays
        - X_missing is None if not provided
    """
    from sklearn.preprocessing import OrdinalEncoder

    n = len(X)

    if numeric_cols:
        X_true = X[numeric_cols].values
        X_imp = X_imputed[numeric_cols].values
        X_miss = X_missing[numeric_cols].values if X_missing is not None else None
    else:
        X_true = np.empty((n, 0))
        X_imp = np.empty((n, 0))
        X_miss = np.empty((n, 0)) if X_missing is not None else None

    if categorical_cols:
        oe = OrdinalEncoder(encoded_missing_value=np.nan, handle_unknown='use_encoded_value', unknown_value=np.nan)

        # Fit on combined data to ensure consistent encoding
        dfs_to_combine = [X[categorical_cols], X_imputed[categorical_cols]]
        if X_missing is not None:
            dfs_to_combine.append(X_missing[categorical_cols])
        combined = pd.concat(dfs_to_combine, ignore_index=True)
        oe.fit(combined)

        encoded_true = oe.transform(X[categorical_cols])
        encoded_imputed = oe.transform(X_imputed[categorical_cols])

        X_true = np.hstack([X_true, encoded_true])
        X_imp = np.hstack([X_imp, encoded_imputed])

        if X_missing is not None:
            encoded_missing = oe.transform(X_missing[categorical_cols])
            X_miss = np.hstack([X_miss, encoded_missing])

    return X_true, X_imp, X_miss


def _evaluate_reconstruction(
    X: pd.DataFrame,
    X_imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Evaluate reconstruction metrics."""
    scores = {}

    if numeric_cols:
        mae_scores = {}
        for col in numeric_cols:
            col_mask = missing_mask[col].values
            if col_mask.any():
                y_true = X[col].values[col_mask]
                y_pred = X_imputed[col].values[col_mask]
                mae_scores[col] = mae(y_true, y_pred)

        if mae_scores:
            mae_scores['avg'] = np.mean(list(mae_scores.values()))
            scores['mae'] = mae_scores

        rmse_scores = {}
        for col in numeric_cols:
            col_mask = missing_mask[col].values
            if col_mask.any():
                y_true = X[col].values[col_mask]
                y_pred = X_imputed[col].values[col_mask]
                rmse_scores[col] = rmse(y_true, y_pred)

        if rmse_scores:
            rmse_scores['avg'] = np.mean(list(rmse_scores.values()))
            scores['rmse'] = rmse_scores

        nrmse_scores = {}
        for col in numeric_cols:
            col_mask = missing_mask[col].values
            if col_mask.any():
                y_true = X[col].values[col_mask]
                y_pred = X_imputed[col].values[col_mask]
                nrmse_scores[col] = nrmse(y_true, y_pred)

        if nrmse_scores:
            nrmse_scores['avg'] = np.mean(list(nrmse_scores.values()))
            scores['nrmse'] = nrmse_scores

    if categorical_cols:
        acc_scores = {}
        for col in categorical_cols:
            col_mask = missing_mask[col].values
            if col_mask.any():
                y_true = X[col].values[col_mask]
                y_pred = X_imputed[col].values[col_mask]
                acc_scores[col] = categorical_accuracy(y_true, y_pred)

        if acc_scores:
            acc_scores['avg'] = np.mean(list(acc_scores.values()))
            scores['categorical_accuracy'] = acc_scores

    return scores


def _evaluate_distribution(
    X: pd.DataFrame,
    X_imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Evaluate distribution metrics."""
    scores = {}

    imputed_numeric_cols = [col for col in numeric_cols if missing_mask[col].any()]
    imputed_categorical_cols = [col for col in categorical_cols if missing_mask[col].any()]

    if imputed_numeric_cols:
        wd_scores = {}
        for col in imputed_numeric_cols:
            true_col = X[col].dropna().values
            imputed_col = X_imputed[col].dropna().values
            if len(true_col) > 0 and len(imputed_col) > 0:
                wd_scores[col] = wasserstein_distance(true_col, imputed_col)

        if wd_scores:
            wd_scores['avg'] = np.mean(list(wd_scores.values()))
            scores['wasserstein_distance'] = wd_scores

        ks_scores = {}
        for col in imputed_numeric_cols:
            true_col = X[col].dropna().values
            imputed_col = X_imputed[col].dropna().values
            if len(true_col) > 0 and len(imputed_col) > 0:
                ks_scores[col] = ks_statistic(true_col, imputed_col)

        if ks_scores:
            ks_scores['avg'] = np.mean(list(ks_scores.values()))
            scores['ks_statistic'] = ks_scores

        kl_scores = {}
        for col in imputed_numeric_cols:
            true_col = X[col].dropna().values
            imputed_col = X_imputed[col].dropna().values
            if len(true_col) > 0 and len(imputed_col) > 0:
                kl_scores[col] = kl_divergence(
                    true_col, imputed_col, feature_type='numerical'
                )

        if kl_scores:
            kl_scores['avg'] = np.mean(list(kl_scores.values()))
            scores['kl_divergence_numeric'] = kl_scores

    if len(numeric_cols) >= 2:
        X_true_arr = X[numeric_cols].values
        X_imputed_arr = X_imputed[numeric_cols].values
        scores['correlation_shift'] = correlation_shift(X_true_arr, X_imputed_arr)

    if imputed_categorical_cols:
        kl_cat_scores = {}
        for col in imputed_categorical_cols:
            true_col = X[col].dropna().values
            imputed_col = X_imputed[col].dropna().values
            if len(true_col) > 0 and len(imputed_col) > 0:
                kl_cat_scores[col] = kl_divergence(
                    true_col, imputed_col, feature_type='categorical'
                )

        if kl_cat_scores:
            kl_cat_scores['avg'] = np.mean(list(kl_cat_scores.values()))
            scores['kl_divergence_categorical'] = kl_cat_scores

    return scores
