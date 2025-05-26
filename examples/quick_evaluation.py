"""Quick example of using the pretty print evaluation summary.

This minimal example shows how to quickly evaluate an imputation method
and display results using the pretty print function.
"""

from imputify.datasets import make_dataset
from imputify import MeanImputer
from imputify.evaluation import evaluate_imputation
from imputify.utils import print_evaluation_summary
import pandas as pd


# Generate data with missing values
X_missing, X_complete, y = make_dataset(
    dataset_type='regression',
    definition={'n_samples': 200, 'n_features': 5},
    amputation={'prop': 0.4}
)

# Apply imputation
imputer = MeanImputer()
df_imputed = pd.DataFrame(
    imputer.fit_transform(X_missing), 
    columns=X_missing.columns
)

print_evaluation_summary(
    results=evaluate_imputation(
        X_true=X_complete,
        X_imputed=df_imputed,
        mask=X_missing.isna()
    ),
    df_missing=X_missing,
    df_complete=X_complete,
    df_imputed=df_imputed,
    method_name="Mean Imputation"
)