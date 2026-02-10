"""Distribution metrics for imputation evaluation.

These metrics compare the statistical distributions of imputed data
against original data. Unlike reconstruction metrics which only evaluate
imputed values, distribution metrics look at entire column distributions.
"""

from typing import Literal, TypeAlias

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein, ks_2samp, entropy

FeatureType: TypeAlias = Literal['categorical', 'numerical']

MIN_HISTOGRAM_BINS = 5
"""Minimum number of bins for adaptive KL divergence histograms."""

MAX_HISTOGRAM_BINS = 50
"""Maximum number of bins for adaptive KL divergence histograms."""


def wasserstein_distance(
    true_col: np.ndarray,
    imputed_col: np.ndarray,
) -> float:
    """Wasserstein distance between two distributions (numerical only).

    Measures the "earth mover's distance" - how much work is needed
    to transform one distribution into another.

    Args:
        true_col: Original column values (full column, not just imputed)
        imputed_col: Imputed column values (full column)

    Returns:
        Distance value (0 = identical, higher = more different)

    Note:
        Both arrays should exclude NaN values before calling.
    """
    if len(true_col) == 0 or len(imputed_col) == 0:
        return float('nan')
    return float(scipy_wasserstein(true_col, imputed_col))


def ks_statistic(
    true_col: np.ndarray,
    imputed_col: np.ndarray,
) -> float:
    """Kolmogorov-Smirnov statistic (numerical only).

    Measures maximum distance between cumulative distributions.
    Tests if two samples come from the same distribution.

    Args:
        true_col: Original column values
        imputed_col: Imputed column values

    Returns:
        KS statistic (0 = identical, 1 = completely different)
    """
    if len(true_col) <= 1 or len(imputed_col) <= 1:
        return float('nan')
    statistic, _ = ks_2samp(true_col, imputed_col)
    return float(statistic)


def _adaptive_bins(n: int) -> int:
    """Choose number of histogram bins based on data size (Sturges-like)."""
    return min(max(int(np.sqrt(n)), MIN_HISTOGRAM_BINS), MAX_HISTOGRAM_BINS)


def kl_divergence(
    true_col: np.ndarray,
    imputed_col: np.ndarray,
    feature_type: FeatureType,
    bins: int | None = None,
    epsilon: float = 1e-10,
) -> float:
    """Kullback-Leibler divergence (both numerical and categorical).

    Measures how one probability distribution differs from another.
    Uses histogram approximation for numerical, exact PMF for categorical.

    Args:
        true_col: Original column values
        imputed_col: Imputed column values
        feature_type: 'numerical' or 'categorical'
        bins: Number of histogram bins for numerical features
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence (0 = identical, higher = more different)
    """
    if feature_type == 'numerical':
        effective_bins = bins if bins is not None else _adaptive_bins(len(true_col))
        return _kl_divergence_numerical(true_col, imputed_col, effective_bins, epsilon)
    else:
        return _kl_divergence_categorical(true_col, imputed_col, epsilon)


def correlation_shift(
    X: np.ndarray,
    X_imputed: np.ndarray,
) -> float:
    """Correlation structure shift (numerical features only).

    Measures how much correlation structure between features changes
    after imputation. Computes normalized Frobenius norm of the
    difference between correlation matrices.

    Args:
        X: Original data matrix (n_samples, n_features)
        X_imputed: Imputed data matrix (n_samples, n_features)

    Returns:
        Normalized shift (0 = no change, higher = more structural change)

    Note:
        Requires at least 2 numerical features. Returns 0.0 otherwise.
    """
    if X.shape[1] < 2:
        return 0.0

    true_corr = np.corrcoef(X.T)
    imputed_corr = np.corrcoef(X_imputed.T)

    # Handle NaN from constant columns — exclude them from comparison
    nan_mask = np.isnan(true_corr) | np.isnan(imputed_corr)
    if nan_mask.all():
        return float('nan')
    true_corr = np.where(nan_mask, 0.0, true_corr)
    imputed_corr = np.where(nan_mask, 0.0, imputed_corr)

    diff = true_corr - imputed_corr
    frob_norm = np.linalg.norm(diff, ord='fro')

    n_features = X.shape[1]
    n_pairs = n_features * (n_features - 1) / 2

    return float(frob_norm / np.sqrt(n_pairs)) if n_pairs > 0 else 0.0



def _kl_divergence_numerical(
    true_col: np.ndarray,
    imputed_col: np.ndarray,
    bins: int = 20,
    epsilon: float = 1e-10,
) -> float:
    """KL divergence for numerical data using histograms."""
    if np.unique(true_col).size <= 1 or np.unique(imputed_col).size <= 1:
        return 0.0

    min_val = min(true_col.min(), imputed_col.min())
    max_val = max(true_col.max(), imputed_col.max())

    if min_val == max_val:
        return 0.0

    hist_true, _ = np.histogram(
        true_col, bins=bins, range=(min_val, max_val), density=True
    )
    hist_imputed, _ = np.histogram(
        imputed_col, bins=bins, range=(min_val, max_val), density=True
    )

    hist_true = (hist_true + epsilon) / (hist_true + epsilon).sum()
    hist_imputed = (hist_imputed + epsilon) / (hist_imputed + epsilon).sum()

    return float(entropy(hist_true, hist_imputed))


def _kl_divergence_categorical(
    true_col: np.ndarray,
    imputed_col: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """KL divergence for categorical data using PMF."""
    all_categories = np.unique(np.concatenate([true_col, imputed_col]))

    if len(all_categories) <= 1:
        return 0.0

    true_counts = {cat: np.sum(true_col == cat) for cat in all_categories}
    imputed_counts = {cat: np.sum(imputed_col == cat) for cat in all_categories}

    true_pmf = np.array([true_counts[cat] for cat in all_categories]) / len(true_col)
    imputed_pmf = np.array([imputed_counts[cat] for cat in all_categories]) / len(
        imputed_col
    )

    true_pmf = (true_pmf + epsilon) / (true_pmf + epsilon).sum()
    imputed_pmf = (imputed_pmf + epsilon) / (imputed_pmf + epsilon).sum()

    return float(entropy(true_pmf, imputed_pmf))
