from __future__ import annotations

from typing import Any

import inspect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from imputify.shared_types import Device, resolve_device
from imputify.imputer.mixins import BaseImputerMixin


class DeviceMixin:
    """Resolves torch device (CUDA -> MPS -> CPU) for imputers."""

    def _resolve_device(self, device: Device = "auto") -> torch.device:
        """Resolve device string to torch.device with automatic fallback.

        Args:
            device: Device specification ('auto', 'cuda', 'mps', 'cpu').

        Returns:
            Resolved torch.device.
        """
        return resolve_device(device)


class TabularDataMixin(BaseImputerMixin):
    """Column splitting and DataLoader creation for tabular neural networks.

    Separates numerical/categorical columns and builds batched
    TensorDatasets with corresponding missing masks.
    """

    _num_cols: list[str]
    _cat_cols: list[str]

    def _split_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Split DataFrame columns into numerical and categorical lists.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (numerical column names, categorical column names).
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return num_cols, cat_cols

    def _create_dataloader(
        self,
        X_processed: dict[str, Any],
        missing_mask: pd.DataFrame,
        batch_size: int,
    ) -> DataLoader:
        """Build a DataLoader from preprocessed data and missing mask.

        Args:
            X_processed: Dict with 'num' (array) and 'cat' (dict of arrays) keys.
            missing_mask: Boolean DataFrame indicating missing positions.
            batch_size: Batch size for the DataLoader.

        Returns:
            Shuffled DataLoader of (X_num, X_cat, mask_num, mask_cat) tensors.
        """
        X_num = torch.FloatTensor(X_processed['num'])
        cat_arrays = [X_processed['cat'][col] for col in self._cat_cols]
        X_cat = torch.LongTensor(np.column_stack(cat_arrays)) if cat_arrays else torch.zeros((X_num.shape[0], 0), dtype=torch.long, device=self._device)
        mask_num = torch.FloatTensor(missing_mask[self._num_cols].values)
        mask_cat = torch.FloatTensor(missing_mask[self._cat_cols].values) if self._cat_cols else torch.zeros((X_num.shape[0], 0), dtype=torch.float32, device=self._device)
        return DataLoader(TensorDataset(X_num, X_cat, mask_num, mask_cat), batch_size=batch_size, shuffle=True)


class SklearnMixin:
    """Sklearn BaseEstimator/TransformerMixin compatibility.

    Provides fit_transform, get_params and set_params
    so imputers work with sklearn pipelines and utilities.
    """

    def fit_transform(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Fit and transform in a single step.

        Args:
            X: Input data.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def _get_param_names(self) -> list[str]:
        """Extract __init__ parameter names via introspection."""
        sig = inspect.signature(self.__init__)
        return [p.name for p in sig.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get imputer parameters as a dict (sklearn convention).

        Args:
            deep: Ignored. Present for sklearn compatibility.

        Returns:
            Dict mapping parameter names to current values.
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> SklearnMixin:
        """Set imputer parameters (sklearn convention).

        Args:
            **params: Parameter names and values to set.

        Returns:
            The imputer instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
