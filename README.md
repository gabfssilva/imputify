# Imputify

A comprehensive framework for evaluating missing data imputation methods.

## Overview

Imputify is a Python library designed to provide a unified framework for evaluating and comparing different missing data imputation techniques. It offers:

- Multiple imputation strategies (statistical, machine learning, stacked)
- Comprehensive evaluation metrics across three dimensions (reconstruction accuracy, distributional fidelity, predictive utility)
- Tools for creating datasets with missing values using multivariate amputation (via pyampute)
- Real-world datasets with missing values
- Comprehensive visualization tools using Plotly
- Easy integration with scikit-learn, pandas, numpy, and other data science tools

## Installation

```bash
pip install imputify
```

## Requirements

- Python >=3.13
- JAX >=0.6.0
- Keras >=3.9.2
- scikit-learn >=1.6.1
- NumPy >=1.26.0
- pandas >=2.0.0
- matplotlib >=3.7.0
- seaborn >=0.12.0
- plotly >=5.18.0
- pyampute >=0.0.3

## Quick Start

```python
import pandas as pd
from imputify.datasets import make_dataset
from imputify import MeanImputer, MedianImputer, MostFrequentImputer
from imputify.methods import KNNImputer, StackedImputer
from imputify.evaluation import evaluate_imputation, compare_imputers
from imputify.utils import print_evaluation_summary

# Generate synthetic data with missing values
df_missing, df_complete = make_dataset(
    dataset_type='regression',
    definition={'n_samples': 500, 'n_features': 10},
    amputation={'prop': 0.3, 'seed': 42}
)

# Initialize imputers
mean_imputer = MeanImputer()
median_imputer = MedianImputer()
knn_imputer = KNNImputer(n_neighbors=5)

# Impute missing values
df_mean = pd.DataFrame(
    mean_imputer.fit_transform(df_missing),
    columns=df_missing.columns
)

# Evaluate a single imputation method
results = evaluate_imputation(
    X_true=df_complete,
    X_imputed=df_mean,
    mask=df_missing.isna()
)

# Pretty print results
print_evaluation_summary(
    results=results,
    df_missing=df_missing,
    df_complete=df_complete,
    df_imputed=df_mean,
    method_name="Mean Imputation"
)

# Compare multiple imputation methods
compare_imputers(
    imputers={
        'Mean': MeanImputer(),
        'Median': MedianImputer(),
        'KNN (k=5)': KNNImputer(n_neighbors=5),
        'Stacked': StackedImputer([
            MedianImputer(),
            MostFrequentImputer()
        ])
    },
    df_missing=df_missing,
    df_complete=df_complete,
    print_summary=True
)
```

## Features

### Imputation Methods

- **Statistical Methods** (wrappers around scikit-learn implementations)
  - `MeanImputer`: Replaces missing values with mean
  - `MedianImputer`: Replaces missing values with median
  - `MostFrequentImputer`: Replaces missing values with mode
  - `ConstantImputer`: Replaces missing values with a constant

- **Machine Learning Methods** (wrappers around scikit-learn implementations)
  - `KNNImputer`: K-Nearest Neighbors imputation

- **Advanced Methods**
  - `StackedImputer`: Applies multiple imputation methods in sequence

### Dataset Utilities

#### Synthetic Data Generation

The `make_dataset` function creates synthetic datasets with missing values:

```python
from imputify.datasets import make_dataset

# Create regression dataset
df_missing, df_complete = make_dataset(
    dataset_type='regression',
    definition={
        'n_samples': 1000,
        'n_features': 20,
        'n_informative': 15,
        'noise': 0.1,
        'random_state': 42
    },
    amputation={
        'prop': 0.3,      # 30% missing values
        'seed': 42
    }
)

# Create classification dataset
df_missing, df_complete = make_dataset(
    dataset_type='classification',
    definition={
        'n_samples': 1000,
        'n_features': 20,
        'n_classes': 3,
        'n_informative': 15,
        'random_state': 42
    },
    amputation={
        'prop': 0.25,
        'seed': 42
    }
)
```

#### Real-world Datasets

Pre-loaded datasets with natural missing values:

```python
from imputify.datasets import (
    load_airquality,
    load_heart_disease,
    load_breast_cancer_wisconsin,
    load_titanic
)

# Load Air Quality dataset
X, y = load_airquality()

# Load Heart Disease dataset
X, y = load_heart_disease()
```

### Missing Data Analysis

```python
from imputify import missing_mask, missing_patterns

# Get binary mask of missing values
mask = missing_mask(df_missing)

# Analyze missing data patterns
patterns = missing_patterns(
    df_missing,
    normalize=True,
    sort_by='count'
)
```

### Evaluation Metrics

#### Reconstruction Accuracy Metrics

Measures how accurately missing values are estimated:

```python
from imputify.metrics import mae, rmse, nrmse, categorical_accuracy

# Mean Absolute Error
mae_score = mae(X_true, X_imputed, mask)

# Root Mean Squared Error
rmse_score = rmse(X_true, X_imputed, mask)

# Normalized RMSE
nrmse_score = nrmse(X_true, X_imputed, mask)
```

#### Distributional Fidelity Metrics

Assesses how well the imputed dataset preserves statistical properties:

```python
from imputify.metrics import kl_divergence, wasserstein_distance, ks_statistic, correlation_shift

# Kullback-Leibler Divergence
kl_score = kl_divergence(X_true[:, 0], X_imputed[:, 0])

# Wasserstein Distance
wd_score = wasserstein_distance(X_true[:, 0], X_imputed[:, 0])

# Kolmogorov-Smirnov Statistic
ks_score = ks_statistic(X_true[:, 0], X_imputed[:, 0])

# Correlation Shift
corr_shift = correlation_shift(X_true, X_imputed)
```

#### Predictive Utility Metrics

Evaluates how imputation affects downstream machine learning models:

```python
from imputify.metrics import evaluate_estimator
from sklearn.linear_model import LogisticRegression

# Evaluate on classification task
results = evaluate_estimator(
    estimator=LogisticRegression(),
    X_train=X_train_imputed,
    X_test=X_test_imputed,
    y_train=y_train,
    y_test=y_test,
    task='classification',
    cv=5
)
```

### Evaluation Functions

#### Single Method Evaluation

```python
from imputify.evaluation import evaluate_imputation

results = evaluate_imputation(
    X_true=df_complete,
    X_imputed=df_imputed,
    mask=df_missing.isna(),
    reconstruction_metrics=['mae', 'rmse', 'nrmse'],
    distribution_metrics=['kl_divergence', 'wasserstein_distance']
)
```

#### Multiple Methods Comparison

```python
from imputify.evaluation import compare_imputers, quick_compare

# Detailed comparison
results = compare_imputers(
    imputers={
        'Mean': MeanImputer(),
        'Median': MedianImputer(),
        'KNN': KNNImputer(n_neighbors=5)
    },
    df_missing=df_missing,
    df_complete=df_complete,
    print_summary=True
)

# Quick comparison with automatic best method selection
best_method, results = quick_compare(
    imputers={...},
    df_missing=df_missing,
    df_complete=df_complete
)
```

### Visualization Tools

Imputify provides comprehensive visualization tools using Plotly:

#### Missing Data Visualization

```python
from imputify.plot import missing_matrix, missing_bars, missing_heatmap

# Visualize missing data pattern as a matrix
fig1 = missing_matrix(df_missing)

# Visualize missing data as bar chart
fig2 = missing_bars(df_missing)

# Visualize correlation between missing values
fig3 = missing_heatmap(df_missing)
```

#### Imputation Comparison Visualization

```python
from imputify.plot import distribution_comparison, scatter_comparison, imputation_comparison

# Compare distributions of original vs. imputed data
fig4 = distribution_comparison(df_original, df_imputed, feature='feature_0')

# Compare feature relationships before and after imputation
fig5 = scatter_comparison(
    df_original,
    df_imputed,
    x_feature='feature_0',
    y_feature='feature_1'
)

# Compare multiple features across imputation methods
fig6 = imputation_comparison(
    df_original,
    df_missing,
    imputed_data_dict  # Dictionary of imputed datasets
)
```

#### Evaluation Visualization

```python
from imputify.plot import (
    reconstruction_metrics,
    distribution_metrics,
    time_comparison,
    metrics_heatmap,
    metric_radar,
    metric_by_missing_rate,
    metric_by_mechanism
)

# Compare reconstruction metrics across methods
fig7 = reconstruction_metrics(results)

# Compare distribution metrics
fig8 = distribution_metrics(results)

# Compare computational efficiency
fig9 = time_comparison(results)

# Heatmap of all metrics
fig10 = metrics_heatmap(results)

# Radar plot for multi-metric comparison
fig11 = metric_radar(results)

# Performance vs missing rate
fig12 = metric_by_missing_rate(results)

# Performance by missingness mechanism
fig13 = metric_by_mechanism(results)
```

### Utility Functions

#### Pretty Print Summaries

```python
from imputify.utils import print_evaluation_summary, print_comparison_summary

# Print single method evaluation
print_evaluation_summary(
    results=results,
    df_missing=df_missing,
    df_complete=df_complete,
    df_imputed=df_imputed,
    method_name="KNN Imputation",
    show_distribution_details=True
)

# Print multiple methods comparison
print_comparison_summary(
    methods_results=results,
    df_missing=df_missing,
    best_by='rmse',
    best_by_category='reconstruction'
)
```

## Examples

The `examples/` directory contains demonstration scripts:

### Quick Evaluation

[quick_evaluation.py](examples/quick_evaluation.py) demonstrates basic usage:
```bash
python examples/quick_evaluation.py
```

### Method Comparison

[compare_methods.py](examples/compare_methods.py) shows how to compare multiple imputation methods:
```bash
python examples/compare_methods.py
```

## Advanced Usage

### Stacked Imputation

Combine multiple imputation strategies:

```python
from imputify.methods import StackedImputer

# Apply median imputation first, then most frequent for remaining values
stacked = StackedImputer([
    MedianImputer(),
    MostFrequentImputer()
])

df_stacked = pd.DataFrame(
    stacked.fit_transform(df_missing),
    columns=df_missing.columns
)
```

### Custom Evaluation Pipeline

```python
# Create missing data with specific patterns
from pyampute import MultivariateAmputation

# Configure amputation with specific patterns
ma = MultivariateAmputation(
    prop=0.3,
    patterns=[
        {'incomplete_vars': [0, 1], 'freq': 0.5},
        {'incomplete_vars': [2, 3, 4], 'freq': 0.5}
    ]
)

df_missing = ma.fit_transform(df_complete)

# Evaluate with custom metrics
results = evaluate_imputation(
    X_true=df_complete,
    X_imputed=df_imputed,
    mask=df_missing.isna(),
    reconstruction_metrics=['mae', 'rmse'],
    distribution_metrics=['wasserstein_distance'],
    per_feature=True
)
```

## Contributing

TODO

## License

MIT License

## Citation

TODO

## Acknowledgments

This library builds upon several excellent packages:
- scikit-learn for base imputation methods
- pyampute for multivariate amputation procedures
- TODO: map others as well