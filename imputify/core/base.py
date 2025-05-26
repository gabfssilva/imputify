"""Base classes for imputation methods."""

from typing import Optional, Dict, Any, Union, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class BaseImputer(Protocol):
    """Base class for all imputation methods."""

    fitted: bool
    params: Dict[str, Any]

    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseImputer":
        """Fit the imputer on the data.

        Args:
            X: Data to fit the imputer on.
            **kwargs: Additional parameters for fitting.

        Returns:
            self: The fitted imputer.
        """
        ...

    def transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Impute missing values in the data.

        Args:
            X: Data with missing values to impute.
            **kwargs: Additional parameters for transformation.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed values.
        """
        ...

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Fit the imputer and impute missing values.

        Args:
            X: Data with missing values to fit and impute.
            **kwargs: Additional parameters for fitting and transformation.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed values.
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the imputer.

        Returns:
            Dict[str, Any]: Parameters of the imputer.
        """
        ...

    def set_params(self, **params) -> "BaseImputer":
        """Set the parameters of the imputer.

        Args:
            **params: Parameters to set.

        Returns:
            self: The imputer with updated parameters.
        """
        ...