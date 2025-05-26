"""Statistical imputation methods."""
from functools import partial

from sklearn.impute import SimpleImputer

MeanImputer = partial(SimpleImputer, strategy='mean')
MedianImputer = partial(SimpleImputer, strategy='median')
ConstantImputer = partial(SimpleImputer, strategy='constant')
MostFrequentImputer = partial(SimpleImputer, strategy='most_frequent')
