"""Example comparing multiple imputation methods using synthetic dataset.

This example demonstrates:
1. Creating a synthetic dataset with missing values
2. Using the compare_imputers function to evaluate multiple methods
3. Displaying the comparison results
4. Using the quick_compare function for automatic best method selection
"""
import pprint

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from imputify.datasets import make_dataset, load_digits, load_wine, load_iris
from imputify import MeanImputer, MedianImputer, MostFrequentImputer, missing_patterns
from imputify.evaluation import compare_imputers
from sklearn.impute import KNNImputer

from imputify.methods import StackedImputer, VAEImputer, TransformerImputer, TabTransformerImputer

X_missing, X_complete, y = make_dataset(
    dataset_type='classification',
    definition={
        'n_samples': 1000,
        'n_features': 12,
        'n_informative': 7,
        'n_redundant': 2,
        'n_clusters_per_class': 3,
        'hypercube': True
    },
    amputation={
        'prop': 0.4,
        'patterns': [{
            'incomplete_vars': list(range(0, 11)),
            'mechanism': 'MCAR'
        }]
    },
    seed=42
)

result = compare_imputers(
    imputers={
        'Mean': MeanImputer(),
        'Median': MedianImputer(),
        'Most Frequent': MostFrequentImputer(),
        'Median&Most Frequent': StackedImputer([
            MedianImputer(),
            MostFrequentImputer()
        ]),
        'KNN (k=5)': KNNImputer(n_neighbors=5),
        'KNN (k=10)': KNNImputer(n_neighbors=10),
        'VAE': VAEImputer(
            layers_config=[
                (0.75, 'leaky_relu'),
                (0.5, 'leaky_relu')
            ],
            latent_dim=4,
            epochs=50,
            batch_size=16,
            learning_rate=0.001,
            verbose=1,
            random_state=42,
            noise_std=0.1,
            validation_split=0.1,
            early_stopping=False,
        ),
        'Transformer': TransformerImputer(
            num_heads=8,
            d_model=64,
            ff_dim=128,
            epochs=20,
            batch_size=8,
            learning_rate=0.001,
            early_stopping=True,
            verbose=1,
            validation_split=0.1,
            random_state=42,
            noise_std=0.2,
        ),
        'TabTransformer': TabTransformerImputer(
            num_heads=2,
            embedding_dim=64,
            transformer_layers=16,
            ff_dim=256,
            epochs=40,
            batch_size=8,
        ),
    },
    df_missing=X_missing,
    df_complete=X_complete,
    print_summary=True,
    y=y,
    estimator=RandomForestClassifier(n_estimators=250, random_state=42),
    random_state=42,
    fail_on_error=True
)
