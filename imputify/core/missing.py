"""Utilities for handling missing data."""

from typing import Union, Optional

import numpy as np
import pandas as pd


def missing_mask(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Create a boolean mask indicating missing values.
    
    Args:
        X: Input data array or DataFrame to check for missing values.
    
    Returns:
        Boolean numpy array where True indicates missing values.
        For pandas objects, uses isna(), for numpy arrays uses isnan().
    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        >>> mask = missing_mask(df)
        >>> mask.shape
        (3, 2)
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isna().to_numpy()
    return np.isnan(X)

def missing_patterns(
    df: pd.DataFrame,
    normalize: bool = True,
    as_proportion: bool = True,
    sort_by: Optional[str] = "count",
    target: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyze missing‐value patterns in a DataFrame, with extended stats.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    normalize : bool, default True
        If True, include a 'proportion' column showing count / total rows.
    as_proportion : bool, default True
        If True and normalize=True, show proportion as float [0–1]; 
        otherwise as percentage [0–100].
    sort_by : str or None, default "count"
        If "count", "proportion", "n_missing_cols", etc., sort descending.
        If None, preserve original pattern order.
    target : pd.Series or single‐column DataFrame, optional
        If provided, compute the mean(target) for each pattern.

    Returns
    -------
    patterns : pd.DataFrame
        Each row is one unique missing‐value pattern:
          - one column per original column, with 1=missing, 0=present
          - 'count' = number of rows exhibiting that pattern
          - optional 'proportion' = count / total rows (float or %)
          - 'n_missing_cols', 'n_present_cols'
          - 'pattern_frac_cols' = n_missing_cols / n_cols
          - 'missing_cells' = count * n_missing_cols
          - 'pct_of_all_missing' = missing_cells / total_missing_cells
          - 'cum_count', 'cum_proportion' (if normalized)
          - 'pattern_id' = "P1", "P2", …
          - 'target_mean' (if target given)
    """

    mask = df.isna().astype(int)
    n_rows, n_cols = mask.shape
    total_missing_cells = mask.values.sum()

    grouped = mask.groupby(list(mask.columns), sort=False)
    summary = grouped.size().reset_index(name="count")

    if normalize:
        if as_proportion:
            summary["proportion"] = summary["count"] / n_rows
        else:
            summary["proportion"] = summary["count"] * 100 / n_rows

    summary["n_missing_cols"] = summary[mask.columns].sum(axis=1)
    summary["n_present_cols"] = n_cols - summary["n_missing_cols"]
    summary["pattern_frac_cols"] = summary["n_missing_cols"] / n_cols

    summary["missing_cells"] = summary["count"] * summary["n_missing_cols"]
    summary["pct_of_all_missing"] = summary["missing_cells"] / total_missing_cells

    if normalize:
        summary["cum_count"]      = summary["count"].cumsum()
        summary["cum_proportion"] = summary["proportion"].cumsum()

    summary.insert(0, "pattern_id", [f"P{i+1}" for i in range(len(summary))])

    if target is not None:
        y = df[target].squeeze()
        target_means = (
            mask
            .assign(_row_idx=mask.index)
            .groupby(list(mask.columns))["_row_idx"]
            .apply(lambda idxs: y.loc[idxs].mean())
            .rename("target_mean")
            .reset_index()
        )
        summary = summary.merge(target_means, on=list(mask.columns), how="left")

    if sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return summary