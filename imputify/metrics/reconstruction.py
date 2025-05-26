"""Reconstruction accuracy metrics for imputation evaluation."""

from typing import Optional, List, Union, Dict, Callable

import numpy as np
import pandas as pd

from imputify.core.missing import missing_mask
from imputify.metrics.types import ReconstructionMetric


def evaluate_reconstruction(
    X_true: pd.DataFrame,
    X_imputed: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    metrics: Optional[List[Union[str, ReconstructionMetric]]] = None,
    categorical_columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate reconstruction quality using multiple metrics.

    Args:
        X_true: Original DataFrame with no missing values.
        X_imputed: DataFrame with imputed values.
        mask: Boolean DataFrame indicating missing values. If None, automatically detected.
        metrics: List of metrics to compute. If None, defaults to ['mae', 'rmse', 'nrmse'].
        categorical_columns: List of column names to treat as categorical.
                           If None, automatically detected.

    Returns:
        Dictionary with metric names as keys and computed values as values.

    Example:
       results = evaluate_reconstruction(
           X_true=df_original,
           X_imputed=df_imputed,
           metrics=['mae', 'rmse', 'categorical_accuracy']
       )
       print(results)
       {'mae': 0.15, 'rmse': 0.23, 'categorical_accuracy': 0.87}
    """
    # Default metrics
    if metrics is None:
        metrics = ['mae', 'rmse', 'nrmse']

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = _detect_categorical_columns(X_true)

    # Create missing mask if not provided
    if mask is None:
        mask = X_true.isna()

    # Separate numeric and categorical columns for processing
    numeric_columns = [col for col in X_true.columns if col not in categorical_columns]

    results = {}

    for metric in metrics:
        metric_name, metric_func = _metric_callable(metric, categorical_columns)

        if metric_name == 'categorical_accuracy':
            # Process categorical columns only
            if categorical_columns:
                cat_true = X_true[categorical_columns].values
                cat_imputed = X_imputed[categorical_columns].values
                cat_mask = mask[categorical_columns].values

                # Convert to column indices for the categorical function
                cat_indices = list(range(len(categorical_columns)))
                results[metric_name] = categorical_accuracy(cat_indices)(cat_true, cat_imputed, cat_mask)
            else:
                results[metric_name] = None
        else:
            if numeric_columns:
                num_true = X_true[numeric_columns].values
                num_imputed = X_imputed[numeric_columns].values
                num_mask = mask[numeric_columns].values

                results[metric_name] = metric_func(num_true, num_imputed, num_mask)
            else:
                results[metric_name] = 0.0  # No error if no numeric columns

    return results

def mae(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Mean Absolute Error (MAE) between true and imputed values.
    
    Args:
        X_true: True values as numpy array.
        X_imputed: Imputed values as numpy array.
        mask: Boolean mask indicating missing values.
            If None, assumes all values in X_imputed were imputed.
    
    Returns:
        float: MAE value.
    """
    if mask is None:
        mask = missing_mask(X_true)
    
    # Ensure arrays are 2D
    if X_true.ndim == 1:
        X_true = X_true.reshape(-1, 1)
        X_imputed = X_imputed.reshape(-1, 1)
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)
    
    # Extract values where mask is True
    true_values = X_true[mask]
    imputed_values = X_imputed[mask]
    
    if len(true_values) == 0:
        return 0.0
    
    return float(np.mean(np.abs(true_values - imputed_values)))

def rmse(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Root Mean Squared Error (RMSE) between true and imputed values.
    
    Args:
        X_true: True values as numpy array.
        X_imputed: Imputed values as numpy array.
        mask: Boolean mask indicating missing values.
            If None, assumes all values in X_imputed were imputed.
    
    Returns:
        float: RMSE value.
    """
    if mask is None:
        mask = missing_mask(X_true)
    
    # Ensure arrays are 2D
    if X_true.ndim == 1:
        X_true = X_true.reshape(-1, 1)
        X_imputed = X_imputed.reshape(-1, 1)
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)
    
    # Extract values where mask is True
    true_values = X_true[mask]
    imputed_values = X_imputed[mask]
    
    if len(true_values) == 0:
        return 0.0
    
    return float(np.sqrt(np.mean((true_values - imputed_values) ** 2)))

def nrmse(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Normalized Root Mean Squared Error (NRMSE) between true and imputed values.
    
    NRMSE is normalized by the range of the true data.
    
    Args:
        X_true: True values as numpy array.
        X_imputed: Imputed values as numpy array.
        mask: Boolean mask indicating missing values.
            If None, assumes all values in X_imputed were imputed.
    
    Returns:
        float: NRMSE value.
    """
    if mask is None:
        mask = missing_mask(X_true)
    
    # Ensure arrays are 2D
    if X_true.ndim == 1:
        X_true = X_true.reshape(-1, 1)
        X_imputed = X_imputed.reshape(-1, 1)
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)
    
    # Calculate RMSE
    rmse_value = rmse(X_true, X_imputed, mask)
    
    # Calculate the range of true data
    true_range = np.nanmax(X_true) - np.nanmin(X_true)
    
    if true_range == 0:
        return 0.0
    
    return float(rmse_value / true_range)

def categorical_accuracy(
    categorical_columns: Optional[List[int]] = None
):
    """
    :param categorical_columns: Indices of categorical columns.
                If None, all columns are considered categorical.
    :return:
    """

    def inner(
        X_true: np.ndarray,
        X_imputed: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate accuracy for categorical variables.

        Args:
            X_true: True values as numpy array.
            X_imputed: Imputed values as numpy array.
            mask: Boolean mask indicating missing values.
                If None, assumes all values in X_imputed were imputed.
        Returns:
            float: Accuracy value.
        """
        nonlocal categorical_columns

        if mask is None:
            mask = missing_mask(X_true)

        # Ensure arrays are 2D
        if X_true.ndim == 1:
            X_true = X_true.reshape(-1, 1)
            X_imputed = X_imputed.reshape(-1, 1)
            if mask.ndim == 1:
                mask = mask.reshape(-1, 1)

        if categorical_columns is None:
            # If not provided, assume all columns are categorical
            categorical_columns = list(range(X_true.shape[1]))

        # Initialize accuracy counter
        correct = 0
        total = 0

        # For each categorical column
        for col in categorical_columns:
            # Get mask for this column
            col_mask = mask[:, col]

            if np.any(col_mask):
                # Extract true and imputed values
                true_values = X_true[col_mask, col]
                imputed_values = X_imputed[col_mask, col]

                # Count correct imputations
                correct += np.sum(true_values == imputed_values)
                total += len(true_values)

        if total == 0:
            return 1.0  # Perfect score if no imputations needed

        return float(correct / total)

    inner.__name__ = 'categorical_accuracy'
    return inner

# Mapping for reconstruction metrics
reconstruction_metrics = {
    'mae': mae,
    'rmse': rmse,
    'nrmse': nrmse,
    'categorical_accuracy': categorical_accuracy()
}


def _detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns using pandas dtype intelligence.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        List of column names that are likely categorical.
    """
    categorical_cols = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        
        if (dtype.name == 'category' or 
            dtype.name == 'object' or
            dtype.name in ['string', 'boolean'] or
            # Low cardinality numeric = likely categorical
            (pd.api.types.is_integer_dtype(dtype) and 
             df[col_name].nunique() / df[col_name].notna().sum() < 0.2)):
            categorical_cols.append(col_name)
    
    return categorical_cols


def _metric_callable(metric: Union[str, ReconstructionMetric], categorical_columns: Optional[List[str]] = None) -> tuple[str, Callable]:
    """Resolve metric name and callable function.
    
    Args:
        metric: Either string name or callable metric function.
        categorical_columns: Column names for categorical accuracy metric.
        
    Returns:
        Tuple of (metric_name, metric_function).
    """
    if isinstance(metric, Callable):
        return metric.__name__, metric
    
    if metric == 'categorical_accuracy':
        return metric, categorical_accuracy(categorical_columns)
    
    if metric in reconstruction_metrics:
        return metric, reconstruction_metrics[metric]
    
    raise ValueError(f'Unknown metric: {metric}')
