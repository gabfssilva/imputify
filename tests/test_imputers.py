"""Sanity checks: each imputer can fit + transform on iris with MCAR missingness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from imputify import (
    DAEImputer,
    GAINImputer,
    KNNImputer,
    StatisticalImputer,
    VAEImputer,
    introduce_missing,
)


@pytest.fixture()
def iris_missing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Iris dataset with 30% MCAR missingness."""
    raw = load_iris(as_frame=True).frame  # type: ignore[union-attr]
    features = raw.drop(columns="target")
    missing, mask = introduce_missing(features, proportion=0.3, seed=42)
    return features, missing, mask


IMPUTERS = [
    pytest.param(KNNImputer(n_neighbors=5), id="knn"),
    pytest.param(StatisticalImputer(numeric_strategy="mean"), id="mean"),
    pytest.param(StatisticalImputer(numeric_strategy="median"), id="median"),
    pytest.param(DAEImputer(hidden_dim=32, epochs=5, seed=42, verbose=False), id="dae"),
    pytest.param(VAEImputer(hidden_dim=32, epochs=5, seed=42, verbose=False), id="vae"),
    pytest.param(GAINImputer(hidden_dim=32, epochs=5, seed=42, verbose=False), id="gain"),
]


@pytest.mark.parametrize("imputer", IMPUTERS)
def test_fit_transform_no_nans(imputer, iris_missing):
    """After fit+transform the result should have zero NaNs."""
    _, missing, _ = iris_missing
    imputer.fit(missing)
    result = imputer.transform(missing)
    assert not pd.DataFrame(result).isnull().any().any(), "Output still contains NaN"


@pytest.mark.parametrize("imputer", IMPUTERS)
def test_shape_preserved(imputer, iris_missing):
    """Output shape must match input shape."""
    _, missing, _ = iris_missing
    imputer.fit(missing)
    result = imputer.transform(missing)
    assert result.shape == missing.shape


@pytest.mark.parametrize("imputer", IMPUTERS)
def test_observed_values_preserved(imputer, iris_missing):
    """Observed (non-missing) values must not be altered."""
    original, missing, _ = iris_missing
    imputer.fit(missing)
    result = imputer.transform(missing)

    result_df = pd.DataFrame(result, columns=missing.columns)
    observed_mask = ~missing.isnull()
    np.testing.assert_allclose(
        result_df.values[observed_mask.values],
        missing.values[observed_mask.values],
        atol=1e-5,
        err_msg="Observed values were modified",
    )
