"""Unified evaluation interface for all imputation metrics."""

from typing import Dict, Optional, Union, TypedDict, List, Callable

from sklearn import clone
from typing_extensions import NotRequired

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from imputify.metrics.reconstruction import mae, rmse, nrmse, categorical_accuracy, evaluate_reconstruction
from imputify.metrics.predictive import evaluate_estimator
from imputify.metrics.distribution import kl_divergence, wasserstein_distance, ks_statistic, correlation_shift, \
    evaluate_distribution
from imputify.metrics.types import ReconstructionMetric, PredictiveMetric

class ImputationResults(TypedDict):
    """Type-safe structure for imputation evaluation results."""
    reconstruction: Dict[str, float]
    distribution: Dict[str, Dict[str, float]]
    predictive: NotRequired[Dict[str, Dict[str, float]]]

def evaluate_imputation(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    estimator: Optional[BaseEstimator] = None,
    reconstruction_metrics: Optional[List[ReconstructionMetric]] = None,
    distribution_metrics: Optional[List[Callable]] = None,
    predictive_metrics: Optional[List[PredictiveMetric]] = None,
    categorical_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    bins: int = 20,
    epsilon: float = 1e-10
) -> ImputationResults:
    """Evaluate imputation quality across all three dimensions.

    This unified function evaluates imputation methods using:
    1. Reconstruction accuracy metrics (how well missing values are estimated)
    2. Distributional fidelity metrics (how well statistical properties are preserved)
    3. Predictive utility metrics (how imputation affects downstream ML tasks)

    Args:
        X_true: Original DataFrame with no missing values.
        X_imputed: DataFrame with imputed values.
        mask: Boolean DataFrame indicating missing values. If None, automatically detected.
        y: Target variable for predictive evaluation. If None, predictive evaluation is skipped.
        estimator: ML estimator for predictive evaluation. If None, predictive evaluation is skipped.
        reconstruction_metrics: List of reconstruction metric functions. If None, uses [mae, rmse, nrmse, categorical_accuracy()].
        distribution_metrics: List of distribution metric functions. If None, uses [kl_divergence(), wasserstein_distance, ks_statistic, correlation_shift].
        predictive_metrics: List of predictive metric functions. If None, uses defaults based on estimator type.
        categorical_columns: List of column names to treat as categorical.  If None, automatically detected.
        test_size: Proportion of data for testing (predictive evaluation).
        random_state: Random seed for reproducibility.
        bins: Number of bins for histogram-based metrics (KL divergence).
        epsilon: Small value to avoid log(0) in KL divergence computation.

    Returns:
        Typed dictionary with evaluation results containing reconstruction, 
        distribution, and optionally predictive evaluation results.

    Example:
        from imputify.metrics import mae, rmse, kl_divergence, wasserstein_distance
        from sklearn.metrics import accuracy_score
        
        # Custom metrics
        results = evaluate_imputation(
            X_true=original_data,
            X_imputed=imputed_data,
            y=target_variable,
            estimator=RandomForestClassifier(),
            reconstruction_metrics=[mae, rmse],
            distribution_metrics=[kl_divergence(), wasserstein_distance],
            predictive_metrics=[accuracy_score]
        )
        
        # Use defaults
        results = evaluate_imputation(
            X_true=original_data,
            X_imputed=imputed_data
        )
    """
    # Validate that imputation was successful (no NaN values remain)
    if X_imputed.isnull().any().any():
        raise ValueError("X_imputed contains NaN values. Imputation was not successful.")
    
    if reconstruction_metrics is None:
        reconstruction_metrics = [mae, rmse, nrmse, categorical_accuracy()]
        
    if distribution_metrics is None:
        distribution_metrics = [kl_divergence(bins=bins, epsilon=epsilon), wasserstein_distance, ks_statistic, correlation_shift]
    
    results: ImputationResults = {
        'reconstruction': evaluate_reconstruction(
            X_true=X_true,
            X_imputed=X_imputed,
            mask=mask,
            metrics=reconstruction_metrics,
            categorical_columns=categorical_columns
        ),
        'distribution': evaluate_distribution(
            X_true=X_true,
            X_imputed=X_imputed,
            metrics=distribution_metrics,
            bins=bins,
            epsilon=epsilon
        )
    }
    
    # Conditionally evaluate predictive utility
    if y is not None and estimator is not None:
        results['predictive'] = evaluate_estimator(
            X_true=X_true,
            X_imputed=X_imputed,
            y=y,
            estimator=estimator,
            metrics=predictive_metrics,
            test_size=test_size,
            random_state=random_state
        )
    
    return results