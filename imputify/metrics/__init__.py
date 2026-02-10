"""Metrics for imputation evaluation.

Three types of metrics:
1. Reconstruction: Compare imputed vs true values at masked positions
2. Distribution: Compare statistical distributions of entire columns
3. Predictive: Compare predictive model performance

Main entry point: evaluate() for complete evaluation across all perspectives.
"""

from .reconstruction import mae, rmse, nrmse, categorical_accuracy

from .distribution import (
    wasserstein_distance,
    ks_statistic,
    kl_divergence,
    correlation_shift,
)

from .predictive import (
    predictive_comparison,
    compute_classification_metrics,
    compute_regression_metrics,
    infer_task_type,
    predictive_r2_delta,
    predictive_accuracy_delta,
)

from .evaluate import evaluate, EvaluationResults

__all__ = [
    'mae',
    'rmse',
    'nrmse',
    'categorical_accuracy',
    'wasserstein_distance',
    'ks_statistic',
    'kl_divergence',
    'correlation_shift',
    'predictive_comparison',
    'compute_classification_metrics',
    'compute_regression_metrics',
    'infer_task_type',
    'predictive_r2_delta',
    'predictive_accuracy_delta',
    'evaluate',
    'EvaluationResults',
]
