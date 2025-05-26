"""Pretty print functions for evaluation summaries."""
from typing import Dict, Optional, Literal, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plottable import Table, ColumnDefinition
from plottable.cmap import normed_cmap
import matplotlib.patches as patches

from imputify.evaluation import ImputationResults


def print_evaluation_summary(
    results: ImputationResults,
    df_missing: pd.DataFrame,
    df_complete: Optional[pd.DataFrame] = None,
    df_imputed: Optional[pd.DataFrame] = None,
    method_name: str = "Imputation",
    show_distribution_details: bool = False
) -> None:
    """Create and display interactive evaluation summary using plottable.
    
    Args:
        results: Dictionary containing evaluation results from evaluate_imputation
        df_missing: DataFrame with missing values
        df_complete: Original complete DataFrame (optional)
        df_imputed: Imputed DataFrame (optional)
        method_name: Name of the imputation method
        show_distribution_details: Whether to show per-feature distribution metrics
    """
    # Create the plottable evaluation summary
    fig, ax = plottable_evaluation_summary(
        results=results,
        df_missing=df_missing,
        df_complete=df_complete,
        df_imputed=df_imputed,
        method_name=method_name
    )
    
    # Display the table
    plt.show(dpi=4)


def print_comparison_summary(
    methods_results: Dict[str, Dict],
    df_missing: pd.DataFrame,
    best_by: str = 'rmse',
    best_by_category: Literal['reconstruction', 'distribution', 'predictive'] = 'reconstruction',
    higher_is_better: bool = False
) -> None:
    """Create and display interactive comparison of multiple imputation methods using plottable.
    
    This function creates a comprehensive table with grouped metrics for better
    organization and visual clarity.
    
    Args:
        methods_results: Dictionary mapping method names to their evaluation results
        df_missing: DataFrame with missing values
        best_by: Metric name to use for selecting the best method (default: 'rmse')
        best_by_category: Category of the metric ('reconstruction', 'distribution', or 'predictive')
        higher_is_better: Whether higher values are better (False for error metrics like RMSE)
    """
    # Create the comprehensive comparison table

    _, _ = plottable_comparison_summary(
        methods_results=methods_results,
        df_missing=df_missing,
        best_by=best_by,
        best_by_category=best_by_category
    )

    plt.show()


# ============================================================================
# PLOTTABLE IMPLEMENTATIONS  
# ============================================================================

def plottable_evaluation_summary(
    results: ImputationResults,
    df_missing: pd.DataFrame,
    df_complete: Optional[pd.DataFrame] = None,
    df_imputed: Optional[pd.DataFrame] = None,
    method_name: str = "Imputation"
) -> tuple[plt.Figure, plt.Axes]:
    """Create evaluation summary using plottable.
    
    Args:
        results: Dictionary containing evaluation results from evaluate_imputation
        df_missing: DataFrame with missing values
        df_complete: Original complete DataFrame (optional)
        df_imputed: Imputed DataFrame (optional) 
        method_name: Name of the imputation method

    Returns:
        tuple: matplotlib Figure and Axes objects
    """
    
    # Prepare reconstruction metrics data
    if 'reconstruction' in results:
        recon_data = []
        for metric, value in results['reconstruction'].items():
            if isinstance(value, (int, float)):
                grade = _get_performance_grade(value, metric)
                score = _get_performance_score(value, metric)
                
                recon_data.append({
                    "Metric": metric,
                    "Value": value,
                    "Grade": grade,
                    "Score": score
                })
        
        if recon_data:
            recon_df = pd.DataFrame(recon_data)
            
            # Create figure and table
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_axis_off()
            
            # Column definitions like WWC example
            col_defs = [
                ColumnDefinition(
                    name="Metric", 
                    title="Metric", 
                    textprops={"ha": "left", "weight": "bold"},
                    width=3.5
                ),
                ColumnDefinition(
                    name="Value", 
                    title="Value", 
                    textprops={"ha": "center", "weight": "bold"},
                    width=2.0,
                    formatter=lambda x: f"{x:.3f}"
                ),
                ColumnDefinition(
                    name="Grade", 
                    title="Grade", 
                    textprops={"ha": "center", "weight": "bold"},
                    width=1.5
                ),
                ColumnDefinition(
                    name="Score", 
                    title="Score", 
                    textprops={"ha": "center"},
                    width=1.5,
                    formatter=lambda x: f"{x:.1f}"
                )
            ]
            
            # Create table like WWC example
            table = Table(
                recon_df,
                column_definitions=col_defs,
                ax=ax,
                textprops={"fontsize": 10},
                row_dividers=True,
                footer_divider=True,
                even_row_color="#f8f9fa"
            )
            
            # Add title
            fig.suptitle(f'{method_name} - Reconstruction Metrics', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            return fig, ax
    
    # Fallback if no reconstruction metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, 'No reconstruction metrics available', 
            ha='center', va='center', fontsize=14)
    ax.set_axis_off()
    return fig, ax


def plottable_comparison_summary(
    methods_results: Dict[str, Dict],
    df_missing: pd.DataFrame,
    best_by: str = 'rmse',
    best_by_category: Literal['reconstruction', 'distribution', 'predictive'] = 'reconstruction'
) -> tuple[plt.Figure, plt.Axes]:
    """Create comparison table using plottable.
    
    Args:
        methods_results: Dictionary mapping method names to their evaluation results
        df_missing: DataFrame with missing values
        best_by: Metric name to use for ranking methods
        best_by_category: Category of the metric to use for ranking
        
    Returns:
        tuple: matplotlib Figure and Axes objects, or None if no data
    """
    
    # Prepare comprehensive data with all metrics
    table_data = []
    
    for method_name, results in methods_results.items():
        row = {"Method": method_name}
        
        # Extract all metrics from all categories
        for category_name, category_metrics in results.items():
            if isinstance(category_metrics, dict):
                for metric_name, metric_data in category_metrics.items():
                    extracted = _extract_metric_value(metric_data, metric_name)
                    # Prefix column names with category for organization
                    for col_name, value in extracted.items():
                        row[f"{category_name}_{col_name}"] = value
        
        table_data.append(row)
    
    if not table_data:
        return None, None
        
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Determine ranking metric and sort
    rank_by = None
    if best_by_category == 'reconstruction' and f"reconstruction_{best_by.replace('_', ' ').title()}" in df.columns:
        rank_by = f"reconstruction_{best_by.replace('_', ' ').title()}"
    elif best_by_category == 'reconstruction' and f"reconstruction_{best_by.upper()}" in df.columns:
        rank_by = f"reconstruction_{best_by.upper()}"
    else:
        # Default to RMSE or first available reconstruction metric
        recon_cols = [col for col in df.columns if col.startswith('reconstruction_')]
        if recon_cols:
            rank_by = recon_cols[0]
    
    # Sort and add ranking
    if rank_by:
        df = df.sort_values(rank_by, ascending=True)  # Lower is better for most metrics
        df = df.reset_index(drop=True)
        
        for i in range(len(df)):
            df.loc[i, "Rank"] = str(i + 1)
        
        # Reorder columns
        cols = ["Rank", "Method"] + [col for col in df.columns if col not in ["Rank", "Method"]]
        df = df[cols]
    
    # Organize columns by score type groups
    display_df = df.copy()
    
    # Group columns by category
    base_cols = ["Rank", "Method"]
    reconstruction_cols = [col for col in df.columns if col.startswith('reconstruction_')]
    distribution_cols = [col for col in df.columns if col.startswith('distribution_')]
    predictive_cols = [col for col in df.columns if col.startswith('predictive_')]
    
    # Reorder columns by groups
    ordered_cols = base_cols + reconstruction_cols + distribution_cols + predictive_cols
    display_df = display_df[ordered_cols]
    
    # Clean up column names (remove category prefix for display)
    column_mapping = {}
    for col in display_df.columns:
        if col.startswith('reconstruction_'):
            column_mapping[col] = col.replace('reconstruction_', '')
        elif col.startswith('distribution_'):
            column_mapping[col] = col.replace('distribution_', '')
        elif col.startswith('predictive_'):
            column_mapping[col] = col.replace('predictive_', '')
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Create figure and table - make it much wider to accommodate all columns
    fig, ax = plt.subplots(figsize=(24, 4))
    ax.set_axis_off()
    
    # Create column definitions with proper groups
    col_defs = []
    
    # Base columns (no group)
    for col in base_cols:
        if col == "Rank":
            col_defs.append(ColumnDefinition(
                name=col,
                title=col,
                textprops={"ha": "center", "weight": "bold"},
                width=0.8
            ))
        elif col == "Method":
            col_defs.append(ColumnDefinition(
                name=col,
                title=col,
                textprops={"ha": "left", "weight": "bold"},
                width=2.5
            ))
    
    # Reconstruction columns
    for i, col in enumerate([c.replace('reconstruction_', '') for c in reconstruction_cols]):
        original_col = reconstruction_cols[i]
        border = "left" if i == 0 else None
        col_defs.append(ColumnDefinition(
            name=col,
            title=col,
            textprops={"ha": "center", "weight": "bold"},
            width=1.3,
            formatter=lambda x: f"{x:.3f}",
            group="Reconstruction",
            border=border
        ))
    
    # Distribution columns  
    for i, col in enumerate([c.replace('distribution_', '') for c in distribution_cols]):
        original_col = distribution_cols[i]
        border = "left" if i == 0 else None
        col_defs.append(ColumnDefinition(
            name=col,
            title=col,
            textprops={"ha": "center", "weight": "bold"},
            width=1.3,
            formatter=lambda x: f"{x:.3f}",
            group="Distribution",
            border=border
        ))
    
    # Predictive columns
    for i, col in enumerate([c.replace('predictive_', '') for c in predictive_cols]):
        original_col = predictive_cols[i]
        border = "left" if i == 0 else None
        col_defs.append(ColumnDefinition(
            name=col,
            title=col,
            textprops={"ha": "center", "weight": "bold"},
            width=1.3,
            formatter=lambda x: f"{x:.3f}",
            group="Predictive",
            border=border
        ))

    display_df = display_df.set_index("Rank")
    
    table = Table(
        display_df,
        column_definitions=col_defs,
        ax=ax,
        textprops={"fontsize": 9},
        row_dividers=True,
        footer_divider=True,
        even_row_color="#f8f9fa"
    )
    
    # Add title
    fig.suptitle('Imputation Methods Comparison', 
                fontsize=16, fontweight='bold', y=0.95)
    
    return fig, ax


# ============================================================================
# GREAT_TABLES IMPLEMENTATIONS (DEPRECATED)
# ============================================================================

def _get_performance_grade(value: float, metric: str, reverse: bool = False) -> str:
    """Convert metric value to letter grade based on predefined thresholds.
    
    Args:
        value: Metric value to grade.
        metric: Name of the metric for threshold lookup.
        reverse: If True, reverse the grade scale (higher values get better grades).
        
    Returns:
        Letter grade from 'A+' (best) to 'D' (worst).
    """
    thresholds = {
        'mae': [0.05, 0.1, 0.2, 0.4],
        'rmse': [0.05, 0.1, 0.2, 0.4], 
        'nrmse': [0.05, 0.1, 0.2, 0.4],
        'kl_divergence': [0.01, 0.05, 0.1, 0.2],
        'wasserstein_distance': [0.05, 0.1, 0.2, 0.4],
        'ks_statistic': [0.05, 0.1, 0.2, 0.4],
        'correlation_shift': [0.01, 0.05, 0.1, 0.2]
    }
    
    grades = ['A+', 'A', 'B', 'C', 'D'] if not reverse else ['D', 'C', 'B', 'A', 'A+']
    thresh = thresholds.get(metric, [0.1, 0.2, 0.4, 0.8])
    
    for i, t in enumerate(thresh):
        if value <= t:
            return grades[i]
    return grades[-1]


def _get_performance_score(value: float, metric: str) -> float:
    """Convert metric value to 0-100 performance score.
    
    Args:
        value: Metric value to convert.
        metric: Name of the metric for appropriate scaling.
        
    Returns:
        Performance score from 0 (worst) to 100 (best).
    """
    # Lower is better for error metrics
    error_metrics = ['mae', 'rmse', 'nrmse', 'kl_divergence', 'wasserstein_distance', 'ks_statistic', 'correlation_shift']
    
    if metric in error_metrics:
        # For error metrics, use inverse scaling with reasonable thresholds
        if metric in ['mae', 'rmse']:
            # Scale based on typical ranges for these metrics
            # Good performance: < 0.5, Poor: > 2.0
            normalized = max(0, min(1, (2.0 - value) / 1.5))
            return normalized * 100
        elif metric == 'nrmse':
            # NRMSE is typically 0-1, good < 0.1, poor > 0.5
            normalized = max(0, min(1, (0.5 - value) / 0.4))
            return normalized * 100
        else:
            # General exponential decay for other metrics
            return min(100, 100 * np.exp(-5 * value))
    else:
        # For accuracy-like metrics, higher is better
        return min(100, value * 100)





def _extract_metric_value(metric_data, metric_name: str):
    """Extract metric value from various data structures.
    
    Args:
        metric_data: The metric data (can be float, dict, or nested dict)
        metric_name: Name of the metric for formatting
        
    Returns:
        Dict with extracted metric columns
    """
    result = {}
    
    if isinstance(metric_data, (int, float)):
        # Simple numeric value
        result[metric_name.replace('_', ' ').title()] = metric_data
    elif isinstance(metric_data, dict):
        # Check if it's a per-feature metric with aggregations
        if '__mean__' in metric_data:
            result[metric_name.replace('_', ' ').title()] = metric_data['__mean__']
        elif all(isinstance(v, dict) for k, v in metric_data.items() if not k.startswith('__')):
            # Nested dict structure (like predictive metrics)
            for sub_metric, sub_value in metric_data.items():
                if isinstance(sub_value, dict):
                    for sub_sub_metric, sub_sub_value in sub_value.items():
                        if isinstance(sub_sub_value, (int, float)):
                            col_name = f"{metric_name.title()} {sub_metric.title()} {sub_sub_metric.title()}"
                            result[col_name] = sub_sub_value
                elif isinstance(sub_value, (int, float)):
                    col_name = f"{metric_name.title()} {sub_metric.title()}"
                    result[col_name] = sub_value
        else:
            # Feature-level dict, compute mean excluding aggregation keys
            values = [v for k, v in metric_data.items() 
                     if not k.startswith('__') and isinstance(v, (int, float))]
            if values:
                result[metric_name.replace('_', ' ').title()] = np.mean(values)
    
    return result


def export_evaluation_report(fig: plt.Figure, filename: str, format: str = "png") -> None:
    """Export plottable evaluation to file.
    
    Args:
        fig: matplotlib Figure object
        filename: Output filename
        format: Export format ('png', 'pdf', 'svg')
    """
    if format.lower() in ["png", "pdf", "svg"]:
        fig.savefig(f"{filename}.{format.lower()}", bbox_inches='tight', dpi=300)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'png', 'pdf', or 'svg'.")
