"""Stacked imputation methods."""

from typing import Union, List, Sequence, Dict, Any

import numpy as np
import pandas as pd

from ..core.base import BaseImputer


class StackedImputer:
    """Impute missing values using a sequence of imputation methods.
    
    This imputer applies multiple imputation methods in sequence, where each method
    operates on the output of the previous method. This allows for combining different
    imputation strategies to potentially achieve better results.
    
    The imputation methods are applied in the order they are provided.
    """

    def __init__(
        self,
        imputers: Sequence[BaseImputer],
        store_intermediates: bool = False,
        **kwargs
    ):
        """Initialize the stacked imputer.

        Args:
            imputers: Sequence of imputer instances to apply in order.
            store_intermediates: Whether to store intermediate results after each imputer.
            **kwargs: Additional parameters.
        """
        if not imputers:
            raise ValueError("At least one imputer must be provided")

        self.imputers = list(imputers)
        self.store_intermediates = store_intermediates
        self._intermediate_results = []
        self._column_names = None
        self.fitted = False
        self.params = kwargs.copy()

    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> "StackedImputer":
        """Fit all imputers in the stack on the data.

        Args:
            X: Data to fit the imputers on.
            **kwargs: Additional parameters.

        Returns:
            self: The fitted imputer.
        """
        # Store column information for DataFrames
        if isinstance(X, pd.DataFrame):
            self._column_names = X.columns
        else:
            self._column_names = None

        # Clear intermediate results
        self._intermediate_results = []

        # Fit each imputer in sequence
        current_data = X
        for i, imputer in enumerate(self.imputers):
            imputer.fit(current_data, **kwargs)

            # Apply the imputer to get the next input data
            # We need this to ensure each imputer is fitted on partially imputed data
            if i < len(self.imputers) - 1:  # Skip transform for the last imputer during fit
                current_data = imputer.transform(current_data, **kwargs)

                # Store intermediate results if requested
                if self.store_intermediates:
                    self._intermediate_results.append(
                        current_data.copy() if hasattr(current_data, 'copy') else current_data)

        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Impute missing values by applying each imputer in sequence.

        Args:
            X: Data with missing values to impute.
            **kwargs: Additional parameters.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed values.
        """
        if not self.fitted:
            raise ValueError("The imputer has not been fitted yet. Call fit() before transform().")

        # Clear intermediate results if storing is enabled
        if self.store_intermediates:
            self._intermediate_results = []

        # Apply each imputer in sequence
        result = X
        for i, imputer in enumerate(self.imputers):
            result = imputer.transform(result, **kwargs)

            # Store intermediate results if requested
            if self.store_intermediates and i < len(self.imputers) - 1:  # Don't store the final result
                self._intermediate_results.append(result.copy() if hasattr(result, 'copy') else result)

        return result

    def get_intermediate_results(self) -> List[Union[np.ndarray, pd.DataFrame]]:
        """Get the intermediate results after each imputation step.
        
        Returns:
            List of intermediate results after each imputer was applied.
            Empty list if store_intermediates was False.
        """
        return self._intermediate_results

    def get_imputer_names(self) -> List[str]:
        """Get the names of imputers in the stack.
        
        Returns:
            List of imputer class names.
        """
        return [imputer.__class__.__name__ for imputer in self.imputers]

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Fit the imputer and impute missing values.

        Args:
            X: Data with missing values to fit and impute.
            **kwargs: Additional parameters for fitting and transformation.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed values.
        """
        return self.fit(X, **kwargs).transform(X, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the imputer.

        Returns:
            Dict[str, Any]: Parameters of the imputer.
        """
        return self.params

    def set_params(self, **params) -> "StackedImputer":
        """Set the parameters of the imputer.

        Args:
            **params: Parameters to set.

        Returns:
            self: The imputer with updated parameters.
        """
        self.params.update(params)
        return self
