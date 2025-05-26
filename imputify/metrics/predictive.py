"""Predictive utility metrics for imputation evaluation."""

from typing import Dict, Optional, Any, List, Union, Callable

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.utils.validation import check_is_fitted

from imputify.metrics.types import Original, Missing, Target, PredictiveMetric

BaseSearchCV = GridSearchCV | RandomizedSearchCV

classification_metrics = {
    'accuracy': accuracy_score,
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
}

regression_metrics = {
    'r2': r2_score,
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}

def evaluate_estimator(
    X_true: Original,
    X_imputed: Missing,
    y: Target,
    estimator: Union[Any, BaseSearchCV],
    metrics: Optional[List[Union[str, PredictiveMetric]]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """Evaluate predictive models on true vs imputed data.
    
    Args:
        X_true: Original dataset with no missing values.
        X_imputed: Dataset with imputed values.
        y: Target variable.
        estimator: Either a sklearn estimator or search object (GridSearchCV/RandomizedSearchCV).
        metrics: List of metrics to compute. If None, default metrics will be used
                based on the estimator type.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
    
    Returns:
        Dict containing metrics for 'true', 'imputed', 'difference', and optionally 'best_params'.
    """
    # Check if it's a search object
    is_search = isinstance(estimator, BaseSearchCV)
    base_estimator = estimator.estimator if is_search else estimator
    is_regressor_model = not _is_classifier(base_estimator)
    
    # Default metrics
    if metrics is None:
        metrics = ['r2', 'mse', 'mae'] if is_regressor_model else ['accuracy', 'precision', 'recall', 'f1']
    
    if is_search:
        return _evaluate_with_search(
            X_true, X_imputed, y, estimator, metrics, test_size, random_state, is_regressor_model
        )
    else:
        return _evaluate_without_search(
            X_true, X_imputed, y, estimator, metrics, test_size, random_state, is_regressor_model
        )


def _evaluate_with_search(
    X_true: Original,
    X_imputed: Missing,
    y: Target,
    search: BaseSearchCV,
    metrics: List[Union[str, PredictiveMetric]],
    test_size: float,
    random_state: Optional[int],
    is_regressor: bool
) -> Dict[str, Dict[str, float]]:
    """Evaluate with hyperparameter search."""
    
    # Split data to avoid leakage
    X_true_train, X_true_test, X_imputed_train, X_imputed_test, y_train, y_test = train_test_split(
        X_true, X_imputed, y, test_size=test_size, random_state=random_state
    )
    
    # Find best parameters using TRUE data only
    search.fit(X_true_train, y_train)
    best_params = search.best_params_
    
    # Train both models with SAME best parameters
    estimator_true = search.estimator.__class__(**best_params)
    estimator_imputed = search.estimator.__class__(**best_params)
    
    estimator_true.fit(X_true_train, y_train)
    estimator_imputed.fit(X_imputed_train, y_train)
    
    # Evaluate on test sets
    y_true_pred = estimator_true.predict(X_true_test)
    y_imputed_pred = estimator_imputed.predict(X_imputed_test)
    
    # Calculate metrics
    results = {
        'true': {},
        'imputed': {},
        'difference': {},
        'best_params': best_params,
        'best_score': search.best_score_
    }
    
    lower_is_better = ['mse', 'mae'] if is_regressor else []
    
    for metric in metrics:
        metric_name, func = _metric_callable(metric)
        is_lower_better = metric_name in lower_is_better
        
        results['true'][metric_name] = float(func(y_test, y_true_pred))
        results['imputed'][metric_name] = float(func(y_test, y_imputed_pred))
        results['difference'][metric_name] = results['true'][metric_name] - results['imputed'][metric_name]
        
        if is_lower_better:
            results['difference'][metric_name] = -results['difference'][metric_name]
    
    return results

def _evaluate_without_search(
    X_true: Original,
    X_imputed: Missing,
    y: Target,
    estimator: Any,
    metrics: List[Union[str, PredictiveMetric]],
    test_size: float,
    random_state: Optional[int],
    is_regressor: bool
) -> Dict[str, Dict[str, float]]:
    """Original evaluation logic without hyperparameter search."""
    # Split data
    X_true_train, X_true_test, X_imputed_train, X_imputed_test, y_train, y_test = train_test_split(
        X_true, X_imputed, y, test_size=test_size, random_state=random_state
    )

    try:
        check_is_fitted(estimator)
        estimator_true = estimator
    except NotFittedError:
        estimator_true = estimator.__class__(**estimator.get_params())
        estimator_true.fit(X_true_train, y_train)

    estimator_imputed = estimator.__class__(**estimator.get_params())
    estimator_imputed.fit(X_imputed_train, y_train)

    y_true_pred = estimator_true.predict(X_true_test)
    y_imputed_pred = estimator_imputed.predict(X_imputed_test)

    results = {
        'true': {},
        'imputed': {},
        'difference': {}
    }

    lower_is_better = ['mse', 'mae'] if is_regressor else []

    for metric in metrics:
        metric_name, func = _metric_callable(metric)
        is_lower_better = metric_name in lower_is_better

        results['true'][metric_name] = float(func(y_test, y_true_pred))
        results['imputed'][metric_name] = float(func(y_test, y_imputed_pred))
        results['difference'][metric_name] = results['true'][metric_name] - results['imputed'][metric_name]

        if is_lower_better:
            results['difference'][metric_name] = -results['difference'][metric_name]

    return results


def _metric_callable(metric: Union[str, PredictiveMetric]) -> tuple[str, Callable]:
    if isinstance(metric, Callable):
        return metric.__name__, metric

    if metric in classification_metrics:
        return metric, classification_metrics[metric]

    if metric in regression_metrics:
        return metric, regression_metrics[metric]

    raise ValueError(f'Unknown metric {metric}')


def _is_classifier(estimator: Any) -> bool:
    """Determine if an estimator is a classifier."""
    return hasattr(estimator, "predict_proba") or hasattr(estimator, "classes_")