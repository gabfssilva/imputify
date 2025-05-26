"""Type definitions for imputation metrics."""

from typing import Dict, Callable, Optional, Union

import numpy as np

type Original = np.ndarray
type Missing = np.ndarray
type Mask = np.ndarray
type Target = np.ndarray
type Predictions = np.ndarray

type ReconstructionMetric = Callable[[Original, Missing, Optional[Mask]], float]
type DistributionMetric = Callable[[Original, Missing], Union[float, Dict[str, float]]]
type PredictiveMetric = Callable[[Predictions, Target], float]

type Reader[I, M] = Callable[[I], M]

type MetricResults = Dict[str, float]
