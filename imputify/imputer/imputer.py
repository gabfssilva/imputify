"""Protocol definitions for imputers."""
from __future__ import annotations

from typing import Protocol, Iterator, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .transformer.pushable_iterator import TrainingStep

ArrayLike = np.ndarray | pd.DataFrame


class Imputer(Protocol):
    """Base protocol for all imputers (sklearn-compatible).

    All imputers accept numpy arrays or pandas DataFrames.
    Input for fit() must contain missing values (NaN).
    transform() preserves observed values and only imputes missing positions.
    """

    def fit(self, X: ArrayLike, **kwargs) -> 'Imputer':
        """Learn imputation parameters from data with missing values.

        Args:
            X: Input data containing NaN values to learn from.

        Returns:
            The fitted imputer instance.
        """
        ...

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Impute missing values using learned parameters.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        ...

    def fit_transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Fit and transform in a single step.

        Args:
            X: Input data containing NaN values.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        ...


class StreamableImputer(Imputer):
    """Imputer that exposes training progress via iterators.

    Used by imputers where fit/transform are long-running
    and benefit from progress tracking.
    """

    def streamed_fit(
        self,
        X: ArrayLike,
        **kwargs,
    ) -> Iterator[TrainingStep]:
        """Fit with progress streaming.

        Args:
            X: Input data containing NaN values to learn from.

        Returns:
            Iterator yielding progress steps during training.
        """
        ...
