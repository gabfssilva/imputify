"""Baseline imputers wrapping sklearn implementations."""
from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class KNNImputer(BaseEstimator, TransformerMixin):
    """KNN Imputer that handles both numerical and categorical features.

    Wraps sklearn's KNNImputer with automatic encoding for categorical columns.
    Supports OneHot or Ordinal encoding strategies.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : str, default='uniform'
        Weight function used in prediction ('uniform' or 'distance').
    categorical_encoder : Literal['onehot', 'ordinal'], default='onehot'
        Encoding strategy for categorical columns.
        - 'onehot': One-hot encoding (recommended for KNN, no ordering assumed)
        - 'ordinal': Ordinal encoding (more compact, assumes implicit ordering)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        categorical_encoder: Literal['onehot', 'ordinal'] = 'onehot',
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.categorical_encoder = categorical_encoder
        self._imputer = None
        self._encoder = None
        self._columns: list[str] | None = None
        self._num_cols: list[str] | None = None
        self._cat_cols: list[str] | None = None
        self._cat_categories: dict | None = None

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Encode DataFrame to numeric array for KNN.

        Numerical columns pass through as-is. Categorical columns are
        encoded via OneHot or Ordinal, with missing values mapped to NaN.

        Args:
            X: Input DataFrame with original column types.

        Returns:
            Numeric array suitable for sklearn KNNImputer.
        """
        parts = []

        if self._num_cols:
            parts.append(X[self._num_cols].values)

        if self._cat_cols and self._encoder is not None:
            cat_data = X[self._cat_cols].astype(str).replace({'nan': '__PLACEHOLDER__', 'None': '__PLACEHOLDER__'})
            cat_encoded = self._encoder.transform(cat_data)

            if self.categorical_encoder == 'onehot':
                row_sums = cat_encoded.sum(axis=1)
                cat_encoded[row_sums == 0] = np.nan
            else:
                cat_encoded = np.where(cat_encoded == -1, np.nan, cat_encoded)

            parts.append(cat_encoded)

        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def _decode(self, X_imputed: np.ndarray, original_df: pd.DataFrame) -> pd.DataFrame:
        """Decode imputed numeric array back to original DataFrame structure.

        For OneHot-encoded categoricals, picks the highest-probability category.
        For Ordinal-encoded, rounds and inverse-transforms.

        Args:
            X_imputed: Imputed numeric array from KNNImputer.
            original_df: Original DataFrame for structure reference.

        Returns:
            DataFrame with imputed values in original types.
        """
        result = original_df.copy()

        idx = 0

        if self._num_cols:
            n_num = len(self._num_cols)
            result[self._num_cols] = X_imputed[:, idx:idx + n_num]
            idx += n_num

        if self._cat_cols and self._encoder is not None:
            if self.categorical_encoder == 'onehot':
                for col in self._cat_cols:
                    categories = self._cat_categories[col]
                    n_cats = len(categories)
                    col_probs = X_imputed[:, idx:idx + n_cats]
                    col_indices = np.argmax(col_probs, axis=1)
                    result[col] = [categories[i] for i in col_indices]
                    idx += n_cats
            else:
                n_cat = len(self._cat_cols)
                cat_imputed = X_imputed[:, idx:idx + n_cat]
                cat_rounded = np.clip(np.round(cat_imputed), 0, None).astype(int)
                cat_decoded = self._encoder.inverse_transform(cat_rounded)
                for i, col in enumerate(self._cat_cols):
                    result[col] = cat_decoded[:, i]

        return result

    def fit(self, X, y=None):
        """Fit encoder and KNN imputer on data with missing values.

        Args:
            X: Input data containing NaN values.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            The fitted imputer.
        """
        from sklearn.impute import KNNImputer as SKLearnKNNImputer
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self._columns = list(X.columns)
        self._num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if self._cat_cols:
            if self.categorical_encoder == 'onehot':
                self._encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore',
                    categories='auto',
                )
            else:
                self._encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                )

            cat_data = X[self._cat_cols].astype(str)
            cat_clean = cat_data.replace({'nan': np.nan, 'None': np.nan}).dropna()
            self._encoder.fit(cat_clean)

            if self.categorical_encoder == 'onehot':
                self._cat_categories = {
                    col: list(cats)
                    for col, cats in zip(self._cat_cols, self._encoder.categories_)
                }

        X_encoded = self._encode(X)

        self._imputer = SKLearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
        )
        self._imputer.fit(X_encoded)

        return self

    def transform(self, X):
        """Impute missing values using fitted KNN.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        return_numpy = isinstance(X, np.ndarray)
        if return_numpy:
            X = pd.DataFrame(X, columns=self._columns)

        X_encoded = self._encode(X)
        X_imputed = self._imputer.transform(X_encoded)
        result = self._decode(X_imputed, X)

        return result.values if return_numpy else result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _CategoricalEncoderMixin:
    """Shared encode/decode logic for imputers that only handle numeric data.

    Supports 'onehot' and 'ordinal' encoding strategies for categorical columns.
    """

    _columns: list[str] | None
    _num_cols: list[str] | None
    _cat_cols: list[str] | None
    _cat_categories: dict | None
    _encoder: OrdinalEncoder | OneHotEncoder | None
    categorical_encoder: Literal['onehot', 'ordinal']

    def _setup_encoder(self, X: pd.DataFrame) -> None:
        self._columns = list(X.columns)
        self._num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self._encoder = None
        self._cat_categories = None

        if not self._cat_cols:
            return

        cat_clean = X[self._cat_cols].astype(str).replace({'nan': np.nan, 'None': np.nan}).dropna()

        match self.categorical_encoder:
            case 'onehot':
                self._encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore', categories='auto',
                )
                self._encoder.fit(cat_clean)
                self._cat_categories = {
                    col: list(cats)
                    for col, cats in zip(self._cat_cols, self._encoder.categories_)
                }
            case 'ordinal':
                self._encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=np.nan,
                )
                self._encoder.fit(cat_clean)

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self._num_cols:
            parts.append(X[self._num_cols].values.astype(float))
        if self._cat_cols and self._encoder is not None:
            cat_data = X[self._cat_cols].astype(str).replace({'nan': '__PLACEHOLDER__', 'None': '__PLACEHOLDER__'})
            cat_encoded = self._encoder.transform(cat_data)
            match self.categorical_encoder:
                case 'onehot':
                    row_sums = cat_encoded.sum(axis=1)
                    cat_encoded[row_sums == 0] = np.nan
                case 'ordinal':
                    cat_encoded = np.where(cat_encoded == -1, np.nan, cat_encoded)
            parts.append(cat_encoded)
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def _decode(self, X_imputed: np.ndarray, original_df: pd.DataFrame) -> pd.DataFrame:
        result = original_df.copy()
        idx = 0
        if self._num_cols:
            n_num = len(self._num_cols)
            result[self._num_cols] = X_imputed[:, idx:idx + n_num]
            idx += n_num
        if self._cat_cols and self._encoder is not None:
            match self.categorical_encoder:
                case 'onehot':
                    for col in self._cat_cols:
                        categories = self._cat_categories[col]
                        n_cats = len(categories)
                        col_probs = X_imputed[:, idx:idx + n_cats]
                        result[col] = [categories[i] for i in np.argmax(col_probs, axis=1)]
                        idx += n_cats
                case 'ordinal':
                    n_cat = len(self._cat_cols)
                    cat_imputed = X_imputed[:, idx:idx + n_cat]
                    cat_rounded = np.clip(np.round(cat_imputed), 0, None).astype(int)
                    cat_decoded = self._encoder.inverse_transform(cat_rounded)
                    for i, col in enumerate(self._cat_cols):
                        result[col] = cat_decoded[:, i]
        return result


class MICEImputer(_CategoricalEncoderMixin, BaseEstimator, TransformerMixin):
    """MICE (Multiple Imputation by Chained Equations) imputer.

    Wraps sklearn's IterativeImputer with BayesianRidge estimator.

    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of imputation rounds.
    sample_posterior : bool, default=True
        Whether to sample from the predictive posterior (stochastic MICE).
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    categorical_encoder : Literal['onehot', 'ordinal'], default='onehot'
        Encoding strategy for categorical columns.
    random_state : int | None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 10,
        sample_posterior: bool = True,
        tol: float = 1e-3,
        categorical_encoder: Literal['onehot', 'ordinal'] = 'onehot',
        random_state: int | None = None,
    ):
        self.max_iter = max_iter
        self.sample_posterior = sample_posterior
        self.tol = tol
        self.categorical_encoder = categorical_encoder
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self._setup_encoder(X)
        X_encoded = self._encode(X)

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            sample_posterior=self.sample_posterior,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._imputer.fit(X_encoded)
        return self

    def transform(self, X):
        return_numpy = isinstance(X, np.ndarray)
        if return_numpy:
            X = pd.DataFrame(X, columns=self._columns)
        X_encoded = self._encode(X)
        X_imputed = self._imputer.transform(X_encoded)
        result = self._decode(X_imputed, X)
        return result.values if return_numpy else result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MissForestImputer(_CategoricalEncoderMixin, BaseEstimator, TransformerMixin):
    """MissForest imputer using IterativeImputer with Random Forest.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the random forest.
    max_iter : int, default=10
        Maximum number of imputation rounds.
    max_features : Literal['sqrt', 'log2'] | float, default='sqrt'
        Number of features per split (Stekhoven & Bühlmann, 2012 default).
    min_samples_leaf : int, default=1
        Minimum samples per leaf node.
    categorical_encoder : Literal['onehot', 'ordinal'], default='onehot'
        Encoding strategy for categorical columns.
    random_state : int | None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        max_features: Literal['sqrt', 'log2'] | float = 'sqrt',
        min_samples_leaf: int = 1,
        categorical_encoder: Literal['onehot', 'ordinal'] = 'onehot',
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.categorical_encoder = categorical_encoder
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self._setup_encoder(X)
        X_encoded = self._encode(X)

        self._imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._imputer.fit(X_encoded)
        return self

    def transform(self, X):
        return_numpy = isinstance(X, np.ndarray)
        if return_numpy:
            X = pd.DataFrame(X, columns=self._columns)
        X_encoded = self._encode(X)
        X_imputed = self._imputer.transform(X_encoded)
        result = self._decode(X_imputed, X)
        return result.values if return_numpy else result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class XGBoostImputer(_CategoricalEncoderMixin, BaseEstimator, TransformerMixin):
    """XGBoost-based imputer using IterativeImputer with XGBRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=4
        Maximum tree depth.
    learning_rate : float, default=0.1
        Boosting learning rate.
    subsample : float, default=0.8
        Row subsampling ratio per tree.
    colsample_bytree : float, default=0.8
        Column subsampling ratio per tree.
    reg_lambda : float, default=1.0
        L2 regularization.
    max_iter : int, default=10
        Maximum number of imputation rounds.
    categorical_encoder : Literal['onehot', 'ordinal'], default='onehot'
        Encoding strategy for categorical columns.
    random_state : int | None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        max_iter: int = 10,
        categorical_encoder: Literal['onehot', 'ordinal'] = 'onehot',
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.categorical_encoder = categorical_encoder
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from xgboost import XGBRegressor

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self._setup_encoder(X)
        X_encoded = self._encode(X)

        self._imputer = IterativeImputer(
            estimator=XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._imputer.fit(X_encoded)
        return self

    def transform(self, X):
        return_numpy = isinstance(X, np.ndarray)
        if return_numpy:
            X = pd.DataFrame(X, columns=self._columns)
        X_encoded = self._encode(X)
        X_imputed = self._imputer.transform(X_encoded)
        result = self._decode(X_imputed, X)
        return result.values if return_numpy else result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StatisticalImputer(BaseEstimator, TransformerMixin):
    """Simple imputer combining mean/median for numeric and mode for categorical.

    Provides a unified interface for basic imputation strategies, handling
    both numerical and categorical columns appropriately.

    Parameters
    ----------
    numeric_strategy : Literal['mean', 'median'], default='mean'
        Strategy for numerical columns.
    """

    def __init__(self, numeric_strategy: Literal['mean', 'median'] = 'mean'):
        self.numeric_strategy = numeric_strategy
        self._num_imputer = None
        self._cat_imputer = None
        self._num_cols: list[str] | None = None
        self._cat_cols: list[str] | None = None
        self._columns: list[str] | None = None

    def fit(self, X, y=None):
        """Fit imputers for numeric and categorical columns.

        Args:
            X: Input data containing NaN values.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            The fitted imputer.
        """
        from sklearn.impute import SimpleImputer as SKLearnSimpleImputer

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self._columns = list(X.columns)
        self._num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if self._num_cols:
            self._num_imputer = SKLearnSimpleImputer(strategy=self.numeric_strategy)
            self._num_imputer.fit(X[self._num_cols])

        if self._cat_cols:
            self._cat_imputer = SKLearnSimpleImputer(strategy='most_frequent')
            self._cat_imputer.fit(X[self._cat_cols])

        return self

    def transform(self, X):
        """Impute missing values using fitted strategies.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        return_numpy = isinstance(X, np.ndarray)
        if return_numpy:
            X = pd.DataFrame(X, columns=self._columns)

        X_out = X.copy()

        if self._num_cols and self._num_imputer:
            X_out[self._num_cols] = self._num_imputer.transform(X[self._num_cols])

        if self._cat_cols and self._cat_imputer:
            X_out[self._cat_cols] = self._cat_imputer.transform(X[self._cat_cols])

        return X_out.values if return_numpy else X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
