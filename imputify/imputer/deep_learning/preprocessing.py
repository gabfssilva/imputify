"""Tabular data preprocessing for neural networks."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MAX_EMBEDDING_DIM = 50
"""Maximum embedding dimension for categorical features."""


class TabularPreprocessor:
    """Handles heterogeneous tabular data preprocessing for neural networks.

    This class manages the transformation pipeline for mixed numerical and
    categorical features, including:
    - Standardization of numerical features
    - Vocabulary building and encoding for categorical features
    - Initial imputation (mean/mode) to handle NaN values
    - Embedding dimension calculation

    The preprocessor maintains separate handling for numerical and categorical
    columns, allowing neural networks to use appropriate representations
    (continuous values vs embeddings/one-hot).
    """

    def __init__(self):
        """Initialize preprocessor."""
        self._fitted: bool = False
        self._num_scaler: StandardScaler | None = None
        self._cat_vocabs: dict[str, dict[str, int]] = {}
        self._cat_inverse_vocabs: dict[str, dict[int, str]] = {}
        self._embedding_dims: dict[str, int] = {}
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._fill_values: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> 'TabularPreprocessor':
        """Fit preprocessor on data.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with potential missing values
        num_cols : list[str]
            List of numerical column names
        cat_cols : list[str]
            List of categorical column names

        Returns
        -------
        self
            Fitted preprocessor
        """
        self._num_cols = num_cols
        self._cat_cols = cat_cols

        if num_cols:
            self._num_scaler = StandardScaler()
            X_num = X[num_cols].copy()
            num_means = X_num.mean()
            self._fill_values.update(num_means.to_dict())
            X_num_filled = X_num.fillna(num_means)
            self._num_scaler.fit(X_num_filled)

        for col in cat_cols:
            unique_values = X[col].dropna().unique()

            vocab = {val: idx + 1 for idx, val in enumerate(unique_values)}
            vocab['__MISSING__'] = 0

            self._cat_vocabs[col] = vocab
            self._cat_inverse_vocabs[col] = {idx: val for val, idx in vocab.items()}

            cardinality = len(unique_values)
            self._embedding_dims[col] = min(MAX_EMBEDDING_DIM, (cardinality + 1) // 2)

            mode_values = X[col].mode()
            if len(mode_values) > 0:
                self._fill_values[col] = mode_values[0]
            else:
                self._fill_values[col] = unique_values[0] if len(unique_values) > 0 else '__MISSING__'

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, Any]:
        """Transform DataFrame to numerical tensors.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with potential missing values

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'num': np.ndarray of shape (n_samples, n_numerical_features)
            - 'cat': Dict[str, np.ndarray] mapping column name to integer indices

        Raises
        ------
        ValueError
            If preprocessor is not fitted
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        result = {}

        if self._num_cols:
            X_num = X[self._num_cols].copy()
            for col in self._num_cols:
                X_num[col] = X_num[col].fillna(self._fill_values[col])
            result['num'] = self._num_scaler.transform(X_num)
        else:
            result['num'] = np.empty((len(X), 0))

        result['cat'] = {}
        for col in self._cat_cols:
            X_cat = X[col].copy()
            X_cat = X_cat.fillna(self._fill_values[col])
            result['cat'][col] = X_cat.map(
                lambda x: self._cat_vocabs[col].get(x, 0)
            ).values.astype(np.int64)

        return result

    def inverse_transform(self, data: dict[str, Any]) -> pd.DataFrame:
        """Transform back to original DataFrame format.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with keys:
            - 'num': np.ndarray of standardized numerical features
            - 'cat': Dict[str, np.ndarray] of categorical indices

        Returns
        -------
        pd.DataFrame
            Dataframe in original format

        Raises
        ------
        ValueError
            If preprocessor is not fitted
        """
        if not self._fitted:
            raise ValueError("Must call fit() before inverse_transform()")

        result = pd.DataFrame()

        if self._num_cols and data['num'].shape[1] > 0:
            X_num_inv = self._num_scaler.inverse_transform(data['num'])
            num_df = pd.DataFrame(X_num_inv, columns=self._num_cols)
            result = pd.concat([result, num_df], axis=1)

        for col in self._cat_cols:
            if col in data['cat']:
                indices = data['cat'][col].astype(int)
                result[col] = [
                    self._cat_inverse_vocabs[col].get(idx, self._fill_values[col])
                    for idx in indices
                ]

        all_cols = self._num_cols + self._cat_cols
        result = result[all_cols]

        return result

    @property
    def num_features(self) -> int:
        """Number of numerical features.

        Returns
        -------
        int
            Count of numerical columns
        """
        return len(self._num_cols)

    @property
    def embedding_info(self) -> dict[str, tuple[int, int]]:
        """Categorical embedding information.

        Returns
        -------
        dict[str, tuple[int, int]]
            Mapping of column name to (vocab_size, embedding_dim)
        """
        return {
            col: (len(self._cat_vocabs[col]), self._embedding_dims[col])
            for col in self._cat_cols
        }

    @property
    def cat_features(self) -> list[str]:
        """List of categorical feature names.

        Returns
        -------
        list[str]
            Categorical column names
        """
        return self._cat_cols

    @property
    def num_columns(self) -> list[str]:
        """List of numerical column names.

        Returns
        -------
        list[str]
            Numerical column names
        """
        return self._num_cols
