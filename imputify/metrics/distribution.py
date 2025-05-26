"""Distributional fidelity metrics for imputation evaluation."""

from typing import Dict, Callable, Optional, List, Union
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats


def _is_categorical_column(col: pd.Series) -> bool:
    """Check if a column should be treated as categorical."""
    return pd.api.types.is_categorical_dtype(col) or pd.api.types.is_object_dtype(col)


def _kl_divergence_categorical(true_col: pd.Series, imputed_col: pd.Series, epsilon: float = 1e-10) -> float:
    """Calculate KL divergence for categorical data using probability mass functions."""
    # Get all unique categories from both columns
    all_categories = set(true_col.dropna()) | set(imputed_col.dropna())
    
    if len(all_categories) <= 1:
        return 0.0
    
    # Calculate probability mass functions (PMFs)
    true_pmf = true_col.value_counts(normalize=True).reindex(all_categories, fill_value=0)
    imputed_pmf = imputed_col.value_counts(normalize=True).reindex(all_categories, fill_value=0)
    
    # Add epsilon to avoid log(0)
    true_pmf = true_pmf + epsilon
    imputed_pmf = imputed_pmf + epsilon
    
    # Normalize after adding epsilon
    true_pmf = true_pmf / true_pmf.sum()
    imputed_pmf = imputed_pmf / imputed_pmf.sum()
    
    # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    kl_div = (true_pmf * np.log(true_pmf / imputed_pmf)).sum()
    return float(kl_div)


def _kl_divergence_continuous(true_col: pd.Series, imputed_col: pd.Series, bins: int = 20, epsilon: float = 1e-10) -> float:
    """Calculate KL divergence for continuous data using histograms."""
    if true_col.nunique() <= 1 or imputed_col.nunique() <= 1:
        return 0.0

    min_val = min(true_col.min(), imputed_col.min())
    max_val = max(true_col.max(), imputed_col.max())
    if min_val == max_val:
        return 0.0

    hist_true, _ = np.histogram(true_col, bins=bins, range=(min_val, max_val), density=True)
    hist_imputed, _ = np.histogram(imputed_col, bins=bins, range=(min_val, max_val), density=True)

    hist_true += epsilon
    hist_imputed += epsilon

    hist_true /= hist_true.sum()
    hist_imputed /= hist_imputed.sum()

    return float(stats.entropy(hist_true, hist_imputed))


def kl_divergence(
    bins: int = 20,
    epsilon: float = 1e-10
) -> Callable[[DataFrame, DataFrame], dict[str, float]]:
    """KL divergence between true and imputed distributions.
    
    Automatically detects column types and uses appropriate calculation:
    - Categorical columns: Uses probability mass functions
    - Continuous columns: Uses histogram-based approach
    """

    def kl_divergence_metric(
        X_true: pd.DataFrame,
        X_imputed: pd.DataFrame
    ) -> Dict[str, float]:
        kl_divs = {}

        for col in X_true.columns:
            true_col = X_true[col].dropna()
            imputed_col = X_imputed[col].dropna()

            if true_col.empty or imputed_col.empty:
                kl_divs[col] = 0.0
                continue

            # Dispatch based on column type
            if _is_categorical_column(true_col):
                kl_divs[col] = _kl_divergence_categorical(true_col, imputed_col, epsilon)
            else:
                kl_divs[col] = _kl_divergence_continuous(true_col, imputed_col, bins, epsilon)

        kl_divs.update(_compute_aggregates(kl_divs))
        return kl_divs

    # Set the function name explicitly
    kl_divergence_metric.__name__ = 'kl_divergence'
    return kl_divergence_metric


def wasserstein_distance(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame
) -> Dict[str, float]:
    """Calculate Wasserstein distance between true and imputed distributions.
    
    The Wasserstein distance measures the minimum cost to transform one distribution
    into another. Lower values indicate better distribution preservation.
    
    Args:
        X_true: Original complete data.
        X_imputed: Data with imputed values.
        
    Returns:
        Dictionary with per-column Wasserstein distances and aggregated statistics
        (mean, median, std). Non-numeric columns return 0.0.
    """
    dists = {}

    for col in X_true.columns:
        true_col = X_true[col].dropna()
        imputed_col = X_imputed[col].dropna()

        if true_col.empty or imputed_col.empty or not pd.api.types.is_numeric_dtype(true_col):
            dists[col] = 0.0
            continue

        try:
            dist = stats.wasserstein_distance(true_col, imputed_col)
            dists[col] = float(dist)
        except Exception:
            dists[col] = 0.0

    dists.update(_compute_aggregates(dists))
    return dists


def ks_statistic(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame
) -> Dict[str, float]:
    """Calculate Kolmogorov-Smirnov statistic between true and imputed distributions.
    
    The KS statistic measures the maximum difference between cumulative distribution
    functions. Lower values indicate better distribution preservation.
    
    Args:
        X_true: Original complete data.
        X_imputed: Data with imputed values.
        
    Returns:
        Dictionary with per-column KS statistics and aggregated statistics
        (mean, median, std). Non-numeric columns return 0.0.
    """
    stats_dict = {}

    for col in X_true.columns:
        true_col = X_true[col].dropna()
        imputed_col = X_imputed[col].dropna()

        if true_col.empty or imputed_col.empty or not pd.api.types.is_numeric_dtype(true_col):
            stats_dict[col] = 0.0
            continue

        try:
            stat, _ = stats.ks_2samp(true_col, imputed_col)
            stats_dict[col] = float(stat)
        except Exception:
            stats_dict[col] = 0.0

    stats_dict.update(_compute_aggregates(stats_dict))
    return stats_dict


def correlation_shift(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame
) -> float:
    """Calculate correlation shift between true and imputed correlation matrices.
    
    Measures the Frobenius norm of the difference between correlation matrices,
    normalized by the number of features. Lower values indicate better preservation
    of feature relationships.
    
    Args:
        X_true: Original complete data.
        X_imputed: Data with imputed values.
        
    Returns:
        Normalized Frobenius norm of correlation matrix difference.
        Returns 0.0 if fewer than 2 features.
    """
    if X_true.shape[1] < 2:
        return 0.0

    true_corr = X_true.corr().fillna(0).to_numpy()
    imputed_corr = X_imputed.corr().fillna(0).to_numpy()

    diff = true_corr - imputed_corr
    frob_norm = np.linalg.norm(diff, ord="fro")

    n_features = X_true.shape[1]
    n_pairs = int(n_features * (n_features - 1) / 2)

    return float(frob_norm / (2 * n_pairs)) if n_pairs > 0 else 0.0


def _compute_aggregates(values: Dict[str, float]) -> Dict[str, float]:
    """Compute aggregate statistics from column-wise metric values."""
    if not values:
        return {"__mean__": 0.0, "__median__": 0.0, "__std__": 0.0, "__range__": 0.0}

    vals = list(values.values())
    aggregates = {
        "__mean__": float(np.mean(vals)),
        "__median__": float(np.median(vals)),
        "__std__": float(np.std(vals)),
        "__range__": float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
    }

    return aggregates


def evaluate_distribution(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame,
    metrics: Optional[List[Union[str, Callable]]] = None,
    bins: int = 20,
    epsilon: float = 1e-10
) -> Dict[str, Dict[str, float]]:
    """Evaluate distributional fidelity using multiple metrics.

    Args:
        X_true: Original DataFrame with no missing values.
        X_imputed: DataFrame with imputed values.
        metrics: List of metrics to compute. If None, defaults to
                ['kl_divergence', 'wasserstein_distance', 'ks_statistic', 'correlation_shift'].
        bins: Number of bins for histogram-based metrics (KL divergence).
        epsilon: Small value to avoid log(0) in KL divergence computation.

    Returns:
        Dictionary with metric names as keys and computed values as dictionaries.
        Each metric returns both column-wise and aggregate statistics.

    Example:
        results = evaluate_distribution(
            X_true=df_original,
            X_imputed=df_imputed,
            metrics=['kl_divergence', 'wasserstein_distance']
        )
        print(results)
        {
            'kl_divergence': {'col1': 0.15, 'col2': 0.23, '__mean__': 0.19, ...},
            'wasserstein_distance': {'col1': 0.05, 'col2': 0.08, '__mean__': 0.065, ...}
        }
    """
    # Default metrics
    if metrics is None:
        metrics = ['kl_divergence', 'wasserstein_distance', 'ks_statistic', 'correlation_shift']

    # Available distribution metrics
    distribution_metrics = {
        'kl_divergence': kl_divergence(bins=bins, epsilon=epsilon),
        'wasserstein_distance': wasserstein_distance,
        'ks_statistic': ks_statistic,
        'correlation_shift': lambda x_true, x_imputed: {'__overall__': correlation_shift(x_true, x_imputed)}
    }

    results = {}

    for metric in metrics:
        metric_name, metric_func = _metric_callable_distribution(metric, distribution_metrics)
        metric_result = metric_func(X_true, X_imputed)
        results[metric_name] = metric_result

    return results


def _metric_callable_distribution(metric: Union[str, Callable], distribution_metrics: Dict[str, Callable]) -> tuple[str, Callable]:
    """Resolve metric name and callable function for distribution metrics.
    
    Args:
        metric: Either string name or callable metric function.
        distribution_metrics: Dictionary of available distribution metrics.
        
    Returns:
        Tuple of (metric_name, metric_function).
    """
    if isinstance(metric, Callable):
        return metric.__name__, metric
    
    if metric in distribution_metrics:
        return metric, distribution_metrics[metric]
    
    raise ValueError(f'Unknown distribution metric: {metric}')
