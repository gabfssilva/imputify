"""Functions for comparing multiple imputation methods."""

from typing import Dict, Optional, Union, List, Tuple, Any, Literal

from sklearn import clone
from typing_extensions import TypedDict
import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from imputify.core.base import BaseImputer
from imputify.evaluation import evaluate_imputation, ImputationResults
from imputify.metrics.types import ReconstructionMetric, PredictiveMetric
from imputify.utils import print_comparison_summary
import warnings

class ImputerResult(TypedDict):
    """Complete results for a single imputer."""
    evaluation: ImputationResults  # Results from evaluate_imputation
    imputed_data: pd.DataFrame
    execution_time: float
    error: Optional[str]

def compare_imputers(
    imputers: Dict[str, Union[BaseImputer, Any]],
    df_missing: pd.DataFrame,
    df_complete: Optional[pd.DataFrame] = None,
    # Evaluation parameters
    mask: Optional[pd.DataFrame] = None,
    y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    estimator: Optional[BaseEstimator] = None,
    # Metric selection
    reconstruction_metrics: Optional[List[ReconstructionMetric]] = None,
    distribution_metrics: Optional[List[Any]] = None,
    predictive_metrics: Optional[List[PredictiveMetric]] = None,
    categorical_columns: Optional[List[str]] = None,
    # Misc
    return_imputed_data: bool = True,
    fail_on_error: bool = False,
    print_summary: bool = True,
    best_by: str = 'rmse',
    best_by_category: Literal['reconstruction', 'distribution', 'predictive'] = 'reconstruction',
    higher_is_better: bool = False,
    random_state: Optional[int] = 42,  # Fixed random state for consistent evaluation
) -> Dict[str, ImputerResult]:
    """
    Compare multiple imputation methods on the same dataset.
    
    This function automates the process of:
    1. Applying multiple imputation methods
    2. Evaluating each method using comprehensive metrics
    3. Collecting timing information
    4. Handling errors gracefully
    
    Args:
        imputers: Dictionary mapping imputer names to imputer instances
        df_missing: DataFrame with missing values to impute
        df_complete: Original complete data for evaluation. If None, only 
                    distribution metrics can be computed
        mask: Boolean DataFrame indicating missing values. If None, auto-detected
        y: Target variable for predictive evaluation
        estimator: ML model for predictive evaluation
        reconstruction_metrics: List of reconstruction metrics to use
        distribution_metrics: List of distribution metrics to use
        predictive_metrics: List of predictive metrics to use
        categorical_columns: List of categorical column names
        return_imputed_data: Whether to include imputed datasets in results
        fail_on_error: Whether to stop if an imputer fails (False = skip failed)
        print_summary: Whether to print comparison summary
        best_by: Metric name to use for selecting the best method (default: 'rmse')
        best_by_category: Category of the metric ('reconstruction', 'distribution', or 'predictive')
        higher_is_better: Whether higher values are better (False for error metrics like RMSE)
        
    Returns:
        Dictionary mapping imputer names to ImputerResult dictionaries containing:
        - evaluation: Results from evaluate_imputation
        - imputed_data: The imputed DataFrame (if return_imputed_data=True)
        - execution_time: Time taken for imputation in seconds
        - error: Error message if imputation failed (None if successful)
        
    Example:
        from imputify import MeanImputer, MedianImputer, KNNImputer
        from imputify.evaluation import compare_imputers

        results = compare_imputers(
            imputers={
                'Mean': MeanImputer(),
                'Median': MedianImputer(),
                'KNN-5': KNNImputer(n_neighbors=5),
                'KNN-10': KNNImputer(n_neighbors=10)
            },
            df_missing=data_with_missing,
            df_complete=original_data,
            print_summary=True
        )
    """
    results = {}
    
    # Auto-detect mask if not provided
    if mask is None:
        mask = df_missing.isna()

    if estimator is not None:
        estimator = clone(estimator)

    for name, imputer in imputers.items():
        start_time = time.time()
        result: ImputerResult = {
            'evaluation': {},
            'imputed_data': pd.DataFrame(),
            'execution_time': 0.0,
            'error': None
        }
        
        try:
            X_imputed = imputer.fit_transform(df_missing)
            
            if isinstance(X_imputed, np.ndarray):
                X_imputed = pd.DataFrame(X_imputed, columns=df_missing.columns)
            
            if df_complete is not None:
                start_time = time.time()

                result['evaluation'] = evaluate_imputation(
                    X_true=df_complete,
                    X_imputed=X_imputed,
                    mask=mask,
                    y=y,
                    estimator=estimator,
                    reconstruction_metrics=reconstruction_metrics,
                    distribution_metrics=distribution_metrics,
                    predictive_metrics=predictive_metrics,
                    categorical_columns=categorical_columns,
                    random_state=random_state
                )

            result['execution_time'] = time.time() - start_time
            
            if return_imputed_data:
                result['imputed_data'] = X_imputed
                
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            
            if fail_on_error:
                raise
            else:
                warnings.warn(f"Imputer '{name}' failed: {str(e)}")
        
        results[name] = result
    
    if print_summary and df_complete is not None:
        valid_results = {
            name: res['evaluation'] 
            for name, res in results.items() 
            if res['error'] is None and res['evaluation']
        }
        
        if valid_results:
            print_comparison_summary(
                valid_results, 
                df_missing,
                best_by=best_by,
                best_by_category=best_by_category,
                higher_is_better=higher_is_better
            )
        else:
            print("No successful imputation results to compare.")
    
    return results


def quick_compare(
    imputers: Dict[str, Union[BaseImputer, Any]],
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    # Evaluation parameters
    mask: Optional[pd.DataFrame] = None,
    y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    estimator: Optional[BaseEstimator] = None,
    # Metric selection
    reconstruction_metrics: Optional[List[ReconstructionMetric]] = None,
    distribution_metrics: Optional[List[Any]] = None,
    predictive_metrics: Optional[List[PredictiveMetric]] = None,
    categorical_columns: Optional[List[str]] = None,
    # Misc
    return_imputed_data: bool = True,
    fail_on_error: bool = False,
    # Summary options
    print_summary: bool = True,
    best_by: str = 'rmse',
    best_by_category: Literal['reconstruction', 'distribution', 'predictive'] = 'reconstruction',
    higher_is_better: bool = False,
    random_state: Optional[int] = 42,  # Fixed random state for consistent evaluation
) -> Tuple[Optional[str], Dict[str, ImputerResult]]:
    """
    Quick comparison with sensible defaults and automatic best method selection.
    
    This is a convenience wrapper around compare_imputers that additionally:
    - Requires df_complete (for meaningful comparison)
    - Automatically selects the best method based on specified metric
    - Returns both the best method name and full results
    
    Args:
        imputers: Dictionary mapping imputer names to imputer instances
        df_missing: DataFrame with missing values to impute
        df_complete: Original complete data for evaluation (required)
        mask: Boolean DataFrame indicating missing values. If None, auto-detected
        y: Target variable for predictive evaluation
        estimator: ML model for predictive evaluation
        reconstruction_metrics: List of reconstruction metrics to use
        distribution_metrics: List of distribution metrics to use
        predictive_metrics: List of predictive metrics to use
        categorical_columns: List of categorical column names
        return_imputed_data: Whether to include imputed datasets in results (default: True)
        fail_on_error: Whether to stop if an imputer fails (default: False)
        print_summary: Whether to print comparison summary (default: True)
        best_by: Metric name to use for selecting the best method (default: 'rmse')
        best_by_category: Category of the metric ('reconstruction', 'distribution', or 'predictive')
        higher_is_better: Whether higher values are better (False for error metrics like RMSE)
    
    Returns:
        Tuple of (best_method_name, full_results_dict)
        - best_method_name: Name of the best imputer based on specified metric, or None if all failed
        - full_results_dict: Complete results from compare_imputers
        
    Example:
        best_method, results = quick_compare(
            imputers={
                'Mean': MeanImputer(),
                'KNN': KNNImputer(n_neighbors=5)
            },
            df_missing=data_with_missing,
            df_complete=original_data
        )
        print(f"Best method: {best_method}")
        print(f"Best RMSE: {results[best_method]['evaluation']['reconstruction']['rmse']}")
    """
    results = compare_imputers(
        imputers=imputers,
        df_missing=df_missing,
        df_complete=df_complete,
        mask=mask,
        y=y,
        estimator=estimator,
        reconstruction_metrics=reconstruction_metrics,
        distribution_metrics=distribution_metrics,
        predictive_metrics=predictive_metrics,
        categorical_columns=categorical_columns,
        return_imputed_data=return_imputed_data,
        fail_on_error=fail_on_error,
        print_summary=print_summary,
        best_by=best_by,
        best_by_category=best_by_category,
        higher_is_better=higher_is_better,
        random_state=random_state,
    )
    
    # Find best method by specified metric
    valid_methods = []
    for name, res in results.items():
        if res['error'] is None and best_by_category in res['evaluation']:
            category_data = res['evaluation'][best_by_category]
            
            # Extract the metric value
            value = None
            if isinstance(category_data, dict):
                if best_by in category_data:
                    value = category_data[best_by]
                    if isinstance(value, dict) and '__mean__' in value:
                        value = value['__mean__']
                else:
                    # Check nested structure
                    for metric_name, metric_data in category_data.items():
                        if metric_name == best_by and isinstance(metric_data, dict) and '__mean__' in metric_data:
                            value = metric_data['__mean__']
                            break
            
            if isinstance(value, (int, float)) and not pd.isna(value):
                valid_methods.append((name, value))
    
    best_method: Optional[str] = None
    if valid_methods:
        # Sort based on higher_is_better
        best_method = sorted(valid_methods, key=lambda x: x[1], reverse=higher_is_better)[0][0]
        print(f"\n🏆 Best method: {best_method}")
    
    return best_method, results