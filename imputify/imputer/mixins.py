from __future__ import annotations

import numpy as np
import pandas as pd


class InputHandlerMixin:
    """Converts input arrays/DataFrames to a consistent DataFrame format
    and infers column types (numerical vs categorical).
    """

    _columns: list[str]

    def _prepare_input_data(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Convert input to DataFrame, reusing stored column names if available.

        Args:
            X: Input data as numpy array or DataFrame.

        Returns:
            Copy of the data as DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if self._columns:
            return pd.DataFrame(X, columns=self._columns)
        return pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])

    def _infer_column_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Classify each column as 'numerical' or 'categorical' by dtype.

        Args:
            df: Input DataFrame.

        Returns:
            Mapping of column name to type string.
        """
        return {
            col: 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
            for col in df.columns
        }


class BaseImputerMixin(InputHandlerMixin):
    """Validation logic shared by all imputers.

    Ensures fit() receives data with NaN, stores column metadata,
    and validates transform() input matches the fitted schema.
    """

    _fitted: bool
    _column_types: dict[str, str]

    def _validate_fit_input(self, X) -> pd.DataFrame:
        """Prepare input and validate it contains missing values.

        Args:
            X: Input data (must contain at least one NaN).

        Returns:
            Prepared DataFrame.

        Raises:
            ValueError: If input has no missing values.
        """
        X_df = self._prepare_input_data(X)
        if not X_df.isnull().any().any():
            raise ValueError("Input data must contain missing values")
        return X_df

    def _setup_columns(self, X_df: pd.DataFrame) -> None:
        """Store column names and inferred types from training data.

        Args:
            X_df: Training DataFrame.
        """
        self._columns = list(X_df.columns)
        self._column_types = self._infer_column_types(X_df)

    def _validate_transform_input(self, X) -> tuple[pd.DataFrame, bool]:
        """Validate input for transform and detect output format.

        Args:
            X: Input data to validate.

        Returns:
            Tuple of (prepared DataFrame, True if input was numpy array).

        Raises:
            ValueError: If not fitted or columns don't match training data.
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")
        return_numpy = isinstance(X, np.ndarray)
        X_df = self._prepare_input_data(X)
        if list(X_df.columns) != self._columns:
            raise ValueError(f"Column mismatch. Expected: {self._columns}, got: {list(X_df.columns)}")
        return X_df, return_numpy
