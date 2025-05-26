"""
imputify: A comprehensive framework for evaluating missing data imputation methods
"""

from .core.base import BaseImputer
from .core.missing import missing_mask, missing_patterns
from .methods.statistical import MeanImputer, MedianImputer, ConstantImputer, MostFrequentImputer
from .methods.knn import KNNImputer
from . import datasets

__version__ = "0.1.0"