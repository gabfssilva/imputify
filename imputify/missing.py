from dataclasses import dataclass
from typing import Literal
import logging

import pandas as pd
from pyampute import MultivariateAmputation

from imputify.shared import global_seed


@dataclass
class PatternConfig:
    """Configuration for a missing data pattern.

    Each pattern defines which variables can have missing values and under
    what mechanism. Multiple patterns can be combined to create complex
    missing data scenarios.

    Args:
        incomplete_vars: List of column names that can have missing values.
        mechanism: Missing data mechanism:
            - 'MCAR': Missing Completely At Random (missingness independent of data)
            - 'MAR': Missing At Random (missingness depends on observed values)
            - 'MNAR': Missing Not At Random (missingness depends on missing value itself)
            - 'MAR+MNAR': Combination of MAR and MNAR
        freq: Relative frequency of this pattern (default 1.0). When using multiple
            patterns, frequencies are normalized to sum to 1.
        weights: Dict mapping column names to weights for computing missingness
            probability. Only used for MAR/MNAR. If None, equal weights are used.
        score_to_probability_func: Function to convert weighted scores to
            missingness probability:
            - 'sigmoid-right': Higher scores → more likely missing
            - 'sigmoid-left': Lower scores → more likely missing
            - 'sigmoid-mid': Middle scores → more likely missing
            - 'sigmoid-tail': Extreme scores → more likely missing

    Example:
        # Simple: make 'age' and 'income' columns MCAR
        PatternConfig(incomplete_vars=['age', 'income'], mechanism='MCAR')

        # MAR: 'income' missing based on 'education' and 'age'
        PatternConfig(
            incomplete_vars=['income'],
            mechanism='MAR',
            weights={'education': 2.0, 'age': 1.0},
            score_to_probability_func='sigmoid-right',
        )

        # Multiple patterns with different frequencies
        patterns = [
            PatternConfig(incomplete_vars=['age'], mechanism='MCAR', freq=0.7),
            PatternConfig(incomplete_vars=['income'], mechanism='MNAR', freq=0.3),
        ]
    """
    incomplete_vars: list[str]
    mechanism: Literal['MCAR', 'MAR', 'MNAR', 'MAR+MNAR'] = 'MCAR'
    freq: float = 1.0
    weights: dict[str, float] | None = None
    score_to_probability_func: Literal[
        'sigmoid-right', 'sigmoid-left', 'sigmoid-mid', 'sigmoid-tail'
    ] | None = None

    def to_dict(self) -> dict:
        """Convert to pyampute pattern dict format."""
        d = {
            'incomplete_vars': self.incomplete_vars,
            'mechanism': self.mechanism,
            'freq': self.freq,
        }
        if self.weights is not None:
            d['weights'] = self.weights
        if self.score_to_probability_func is not None:
            d['score_to_probability_func'] = self.score_to_probability_func
        return d


def introduce_missing(
    df: pd.DataFrame,
    proportion: float = 0.3,
    patterns: list[PatternConfig] | list[dict] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Introduce missing values into a DataFrame using pyampute.

    Args:
        df: Complete DataFrame without missing values.
        proportion: Proportion of incomplete cases (rows with ≥1 missing value).
            Default: 0.3 (30% of rows will have missing values).
        patterns: List of missing data patterns. Can be:
            - List of PatternConfig objects (recommended, type-safe)
            - List of dicts (pyampute format, for backwards compatibility)
            - None: uses simple MCAR on all columns
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (df_missing, mask) where:
            - df_missing: DataFrame with missing values introduced
            - mask: Boolean DataFrame with True where values are missing

    Example:
        # Using PatternConfig (recommended)
        patterns = [
            PatternConfig(incomplete_vars=['age', 'income'], mechanism='MCAR'),
            PatternConfig(incomplete_vars=['education'], mechanism='MAR', freq=0.3),
        ]
        df_missing, mask = introduce_missing(df, proportion=0.3, patterns=patterns)
    """
    seed = seed or global_seed()

    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Input DataFrame already contains NaN values in columns: {nan_cols}")

    cat_columns = df.select_dtypes(include=["category", "object"]).columns.tolist()
    cat_mappings: dict[str, pd.Categorical] = {}

    df_numeric = df.copy()
    for col in cat_columns:
        cat = pd.Categorical(df[col])
        cat_mappings[col] = cat
        df_numeric[col] = cat.codes.astype(float)
        df_numeric.loc[df_numeric[col] < 0, col] = float("nan")

    kwargs = {"prop": proportion, "seed": seed}
    if patterns is not None:
        pattern_dicts = [
            p.to_dict() if isinstance(p, PatternConfig) else p
            for p in patterns
        ]
        kwargs["patterns"] = pattern_dicts

    logging.disable(logging.WARNING)
    try:
        ma = MultivariateAmputation(**kwargs)
        df_amputed = pd.DataFrame(
            ma.fit_transform(df_numeric),
            columns=df.columns,
            index=df.index,
        )
    finally:
        logging.disable(logging.NOTSET)

    mask = df_amputed.isna()

    df_missing = df_amputed.copy()
    for col, cat in cat_mappings.items():
        codes = df_amputed[col].copy()
        codes_int = codes.fillna(-1).astype(int)
        df_missing[col] = pd.Categorical.from_codes(codes_int, categories=cat.categories)
        df_missing.loc[mask[col], col] = None

    return df_missing, mask
