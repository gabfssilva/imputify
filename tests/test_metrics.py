"""Tests for imputify.metrics — reconstruction, distribution, predictive, evaluate."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

from imputify.metrics.reconstruction import mae, rmse, nrmse, categorical_accuracy
from imputify.metrics.distribution import (
    wasserstein_distance,
    ks_statistic,
    kl_divergence,
    correlation_shift,
)
from imputify.metrics.predictive import (
    predictive_comparison,
    infer_task_type,
    compute_classification_metrics,
    compute_regression_metrics,
)
from imputify.metrics.evaluate import evaluate, EvaluationResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=["sl", "sw", "pl", "pw"])
    y = data.target
    return X, y


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

class TestReconstruction:
    def test_perfect_imputation(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0
        assert rmse(y, y) == 0.0

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(12.5))

    def test_nrmse_normalized(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([0.0, 10.0])
        assert nrmse(y_true, y_pred) == 0.0

    def test_nrmse_constant_column_returns_nan(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.1, 5.0])
        assert math.isnan(nrmse(y_true, y_pred))

    def test_categorical_accuracy_perfect(self):
        y = np.array(["a", "b", "c"])
        assert categorical_accuracy(y, y) == 1.0

    def test_categorical_accuracy_partial(self):
        y_true = np.array(["a", "b", "c", "d"])
        y_pred = np.array(["a", "x", "c", "x"])
        assert categorical_accuracy(y_true, y_pred) == pytest.approx(0.5)

    @pytest.mark.parametrize("fn", [mae, rmse, nrmse, categorical_accuracy])
    def test_empty_arrays_return_nan(self, fn):
        assert math.isnan(fn(np.array([]), np.array([])))


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

class TestDistribution:
    def test_wasserstein_identical(self, rng):
        x = rng.randn(200)
        assert wasserstein_distance(x, x) == pytest.approx(0.0)

    def test_wasserstein_shifted(self):
        a = np.zeros(100)
        b = np.ones(100)
        assert wasserstein_distance(a, b) == pytest.approx(1.0)

    def test_ks_identical(self, rng):
        x = rng.randn(200)
        assert ks_statistic(x, x) == pytest.approx(0.0)

    def test_ks_different(self):
        a = np.zeros(100)
        b = np.ones(100)
        assert ks_statistic(a, b) == pytest.approx(1.0)

    def test_ks_too_few_returns_nan(self):
        assert math.isnan(ks_statistic(np.array([1.0]), np.array([2.0])))

    def test_kl_identical_numerical(self, rng):
        x = rng.randn(500)
        kl = kl_divergence(x, x, feature_type="numerical")
        assert kl < 0.01  # should be ~0

    def test_kl_different_numerical(self, rng):
        a = rng.randn(500)
        b = rng.randn(500) + 5.0
        kl = kl_divergence(a, b, feature_type="numerical")
        assert kl > 0.1

    def test_kl_identical_categorical(self):
        x = np.array(["a", "b", "c"] * 100)
        kl = kl_divergence(x, x, feature_type="categorical")
        assert kl < 0.01

    def test_kl_different_categorical(self):
        a = np.array(["a"] * 100)
        b = np.array(["b"] * 100)
        kl = kl_divergence(a, b, feature_type="categorical")
        assert kl > 0.1

    def test_correlation_shift_identical(self, rng):
        X = rng.randn(100, 4)
        assert correlation_shift(X, X) == pytest.approx(0.0)

    def test_correlation_shift_perturbed(self, rng):
        X = rng.randn(100, 4)
        X_noisy = X + rng.randn(100, 4) * 2
        shift = correlation_shift(X, X_noisy)
        assert shift > 0.0

    def test_correlation_shift_single_feature(self, rng):
        X = rng.randn(50, 1)
        assert correlation_shift(X, X) == 0.0

    def test_wasserstein_empty_returns_nan(self):
        assert math.isnan(wasserstein_distance(np.array([]), np.array([1.0])))


# ---------------------------------------------------------------------------
# Predictive metrics
# ---------------------------------------------------------------------------

class TestPredictive:
    def test_infer_classification(self):
        assert infer_task_type(np.array([0, 1, 2, 0, 1])) == "classification"

    def test_infer_regression(self):
        y = np.linspace(0, 100, 50)
        assert infer_task_type(y) == "regression"

    def test_infer_string_is_classification(self):
        y = np.array(["cat", "dog", "cat"])
        assert infer_task_type(y) == "classification"

    def test_classification_metrics_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        m = compute_classification_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["f1"] == pytest.approx(1.0)

    def test_regression_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_regression_metrics(y, y)
        assert m["r2"] == 1.0
        assert m["mae"] == 0.0

    def test_predictive_comparison_classification(self, iris_data):
        X, y = iris_data
        X_arr = X.values
        results = predictive_comparison(
            X_arr, X_arr, y, task_type="classification", n_folds=3, seed=42,
        )
        assert "X" in results and "X_imputed" in results
        assert results["X"]["accuracy"]["mean"] > 0.8

    def test_predictive_comparison_regression(self):
        data = load_diabetes()
        X, y = data.data, data.target
        results = predictive_comparison(
            X, X, y, task_type="regression", n_folds=3, seed=42,
        )
        assert results["X"]["r2"]["mean"] > 0.0

    def test_predictive_comparison_no_features(self):
        X_empty = np.empty((50, 0))
        y = np.arange(50)
        assert predictive_comparison(X_empty, X_empty, y) is None


# ---------------------------------------------------------------------------
# End-to-end evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_perfect_imputation(self, iris_data):
        X, y = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        # Manually set some positions as "missing" to evaluate
        mask.iloc[0:10, 0] = True
        mask.iloc[5:15, 2] = True

        # Perfect imputation = X itself
        results = evaluate(X=X, X_imputed=X, missing_mask=mask)

        assert isinstance(results, EvaluationResults)
        assert results.overall_reconstruction_score == 0.0
        assert results.predictive is None

    def test_evaluate_with_noise(self, iris_data, rng):
        X, y = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        mask.iloc[0:30, 0] = True
        mask.iloc[20:50, 1] = True

        X_noisy = X.copy()
        for col in X.columns:
            col_mask = mask[col].values
            if col_mask.any():
                X_noisy.loc[col_mask, col] += rng.randn(col_mask.sum()) * 0.5

        results = evaluate(X=X, X_imputed=X_noisy, missing_mask=mask)

        assert results.overall_reconstruction_score > 0.0
        assert "mae" in results.reconstruction
        assert "rmse" in results.reconstruction

    def test_evaluate_with_predictive(self, iris_data):
        X, y = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        mask.iloc[0:10, 0] = True

        results = evaluate(X=X, X_imputed=X, missing_mask=mask, y=y)

        assert results.predictive is not None
        assert results.overall_predictive_score is not None
        assert results.overall_predictive_std is not None

    def test_to_dict_keys(self, iris_data):
        X, y = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        mask.iloc[0:10, 0] = True

        results = evaluate(X=X, X_imputed=X, missing_mask=mask, y=y)
        d = results.to_dict()

        assert "reconstruction" in d
        assert "distribution" in d
        assert "predictive" in d
        assert "summaries" in d
        assert "overall" in d["summaries"]

    def test_overall_score_range(self, iris_data, rng):
        X, _ = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        mask.iloc[0:20, :] = True

        X_noisy = X.copy()
        for col in X.columns:
            col_mask = mask[col].values
            if col_mask.any():
                X_noisy.loc[col_mask, col] += rng.randn(col_mask.sum())

        results = evaluate(X=X, X_imputed=X_noisy, missing_mask=mask)
        assert 0.0 <= results.overall_score <= 1.0

    def test_evaluate_no_missing(self, iris_data):
        X, _ = iris_data
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        results = evaluate(X=X, X_imputed=X, missing_mask=mask)
        # No missing → empty reconstruction/distribution
        assert results.overall_reconstruction_score == 0.0

    def test_evaluation_results_dataclass(self):
        r = EvaluationResults(
            reconstruction={"mae": {"a": 0.1, "avg": 0.1}},
            distribution={"ks_statistic": {"a": 0.05, "avg": 0.05}},
        )
        assert r.overall_reconstruction_score == pytest.approx(0.1)
        assert r.overall_distribution_score == pytest.approx(0.05)
        assert r.predictive is None
        assert r.overall_predictive_score is None
