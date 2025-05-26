"""K-Nearest Neighbors based imputation methods."""
from functools import partial

from sklearn.impute import KNNImputer as SklearnKNNImputer

KNNImputer = partial(SklearnKNNImputer)
