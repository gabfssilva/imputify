from typing import Tuple, Dict, Any, Optional, Literal, TypedDict, cast, overload, Union, List, Callable
import numpy as np
import pandas as pd
import pyampute.utils
from sklearn.datasets import make_classification, make_regression, make_multilabel_classification
from pyampute.ampute import MultivariateAmputation
from scipy.sparse import spmatrix

class Classification(TypedDict, total=False):
    """Configuration for generating synthetic classification datasets.
    
    This TypedDict defines parameters for sklearn.datasets.make_classification,
    which creates a random n-class classification problem with controllable
    complexity and separability.
    
    Attributes:
        n_samples: Number of samples to generate (default: 100).
            Controls dataset size. Larger values create more training data
            but increase computational cost.
            
        n_features: Total number of features (default: 20).
            Includes informative, redundant, and random features.
            Higher values increase dimensionality and potential complexity.
            
        n_informative: Number of informative features (default: 2).
            Features that are actually useful for classification.
            Should be ≤ n_features. Controls signal strength in the data.
            
        n_redundant: Number of redundant features (default: 2).
            Linear combinations of informative features + noise.
            Adds multicollinearity, testing imputers' ability to handle
            correlated features with missing values.
            
        n_repeated: Number of duplicated features (default: 0).
            Exact copies of randomly selected features.
            Tests imputers' robustness to completely redundant information.
            
        n_classes: Number of classes/labels (default: 2).
            For binary classification use 2, for multiclass use >2.
            Higher values increase classification complexity.
            
        n_clusters_per_class: Number of clusters per class (default: 2).
            Controls within-class structure. Higher values create
            more complex decision boundaries and class distributions.
            
        weights: Proportions of samples assigned to each class.
            None for balanced classes, list of floats for imbalanced.
            Imbalanced datasets test imputers under realistic conditions.
            
        flip_y: Fraction of samples with randomly flipped labels (default: 0.01).
            Introduces label noise. Higher values make classification harder
            and test imputation robustness to noisy targets.
            
        class_sep: Multiplicative factor for class separation (default: 1.0).
            Higher values make classes more separable.
            Lower values create overlapping classes, increasing difficulty.
            
        hypercube: If True, place clusters on hypercube vertices (default: True).
            Creates geometrically structured class distributions.
            False generates Gaussian clusters.
            
        shift: Shift features by specified value (default: 0.0).
            Translates the entire feature space. Useful for testing
            imputers' invariance to feature scaling and centering.
            
        scale: Multiply features by specified value (default: 1.0).
            Scales the feature space. Tests imputers' sensitivity
            to feature magnitudes and normalization needs.
            
        shuffle: Whether to shuffle samples and features (default: True).
            Randomizes order to avoid artifacts from generation process.
            Should generally be True for realistic datasets.
            
        random_state: Seed for random number generation.
            Ensures reproducible dataset generation across runs.
            Critical for consistent evaluation and debugging.
    """
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int
    n_repeated: int
    n_classes: int
    n_clusters_per_class: int
    weights: list
    flip_y: float
    class_sep: float
    hypercube: bool
    shift: float
    scale: float
    shuffle: bool
    random_state: int

class Regression(TypedDict, total=False):
    """Configuration for generating synthetic regression datasets.
    
    This TypedDict defines parameters for sklearn.datasets.make_regression,
    which creates a random regression problem with controllable noise,
    dimensionality, and target complexity.
    
    Attributes:
        n_samples: Number of samples to generate (default: 100).
            Controls dataset size. More samples provide better statistical
            power for evaluating imputation quality in regression tasks.
            
        n_features: Total number of input features (default: 100).
            Higher dimensionality tests imputers' ability to handle
            high-dimensional regression with sparse signal.
            
        n_informative: Number of informative features (default: 10).
            Features used to build the linear model for targets.
            Should be ≤ n_features. Controls signal-to-noise ratio
            and how much useful information is available.
            
        n_targets: Number of regression targets (default: 1).
            For single-output regression use 1, for multi-output use >1.
            Multi-output regression tests imputers' ability to handle
            correlated target variables.
            
        noise: Standard deviation of Gaussian noise (default: 0.0).
            Controls observation noise level. Higher values make
            regression harder and test imputers' robustness to noisy data.
            Realistic datasets typically have noise > 0.
            
        tail_strength: Relative importance of tail features (default: 0.5).
            Controls how informative the non-informative features are.
            0.0 makes them pure noise, 1.0 makes them as important
            as informative features. Tests handling of weak signals.
            
        bias: Bias/intercept term added to targets (default: 0.0).
            Shifts target distribution. Tests imputers' ability to
            handle targets with different scales and ranges.
            
        effective_rank: Approximate rank of input-output relationship.
            Controls intrinsic dimensionality of the regression problem.
            Lower values create more structured, lower-rank relationships
            that may be easier for some imputers to model.
            
        coef: Whether to return coefficient matrix (default: False).
            If True, also returns true coefficients used to generate
            targets. Useful for analyzing how well imputation preserves
            true underlying relationships.
            
        shuffle: Whether to shuffle samples and features (default: True).
            Randomizes order to remove artifacts from generation.
            Should generally be True for realistic evaluation.
            
        random_state: Seed for random number generation.
            Ensures reproducible regression datasets across runs.
            Essential for consistent benchmarking and debugging.
    """
    n_samples: int
    n_features: int
    n_informative: int
    n_targets: int
    noise: float
    tail_strength: float
    bias: float
    effective_rank: int
    coef: bool
    shuffle: bool
    random_state: int

class MultilabelClassification(TypedDict, total=False):
    """Configuration for generating synthetic multilabel classification datasets.
    
    This TypedDict defines parameters for sklearn.datasets.make_multilabel_classification,
    which creates random multilabel classification problems where each sample
    can belong to multiple classes simultaneously.
    
    Attributes:
        n_samples: Number of samples to generate (default: 100).
            Controls dataset size. Multilabel problems often need more
            samples than single-label due to increased label complexity.
            
        n_features: Number of input features (default: 20).
            Feature dimensionality. Higher values increase complexity
            and test imputers' ability to handle high-dimensional
            multilabel problems with missing feature values.
            
        n_classes: Number of possible classes (default: 5).
            Total number of distinct labels available. Each sample
            can have any subset of these classes. Higher values
            create more complex label spaces and correlations.
            
        n_labels: Average number of labels per sample (default: 2).
            Controls label density. Higher values mean samples have
            more simultaneous labels, creating richer multilabel
            structure but also more complexity for imputation.
            
        length: Sum of features over all documents (default: 50).
            For bag-of-words representation, controls total vocabulary
            usage across dataset. Affects sparsity and feature
            co-occurrence patterns that imputers must preserve.
            
        allow_unlabeled: Whether to allow samples with no labels (default: True).
            Real multilabel datasets often have unlabeled samples.
            Tests imputers' robustness to extreme label sparsity
            and samples with missing label information.
            
        sparse: Whether to return sparse feature matrix (default: False).
            Sparse features are common in multilabel problems (e.g., text).
            Tests imputers' ability to handle sparse data structures
            and maintain sparsity patterns after imputation.
            
        return_indicator: Format for label matrix ('dense' or 'sparse').
            Controls output format of multilabel targets.
            'sparse' format tests imputers that must handle
            sparse target structures in addition to sparse features.
            
        return_distributions: Whether to return label frequency info (default: False).
            If True, returns additional information about class
            distributions. Useful for analyzing how imputation
            affects label balance and co-occurrence patterns.
            
        random_state: Seed for random number generation.
            Ensures reproducible multilabel datasets. Critical for
            consistent evaluation of imputation methods on complex
            multilabel structures.
    """
    n_samples: int
    n_features: int
    n_classes: int
    n_labels: int
    length: int
    allow_unlabeled: bool
    sparse: bool
    return_indicator: Literal['dense', 'sparse']
    return_distributions: bool
    random_state: int

class Pattern(TypedDict, total=False):
    """Configuration for a single missing data pattern.
    
    This TypedDict defines parameters for pyampute.ampute.MultivariateAmputation
    pattern specifications, which control how missing values are systematically
    introduced into complete datasets following specific missingness mechanisms.
    
    Attributes:
        incomplete_vars: Variables that will have missing values in this pattern.
            Can be list of column indices (int) or column names (str).
            Defines which features will be amputed according to this pattern.
            Different patterns can target different subsets of variables.
            
        weights: Importance weights for calculating missingness probabilities.
            - List[float]: Weights for all variables in order
            - Dict[int, float]: Weights indexed by column position  
            - Dict[str, float]: Weights indexed by column name
            Controls which variables influence whether values become missing.
            Higher weights mean variables have more influence on missingness.
            
        mechanism: Type of missingness mechanism to implement.
            - 'MCAR': Missing Completely At Random - missingness independent of data
            - 'MAR': Missing At Random - missingness depends on observed values
            - 'MNAR': Missing Not At Random - missingness depends on unobserved values
            - 'MAR+MNAR': Combination of MAR and MNAR mechanisms
            Different mechanisms test imputers under different assumptions.
            
        freq: Relative frequency of this pattern (default: 1.0).
            When multiple patterns are used, controls how often each
            pattern is applied relative to others. Higher values make
            this pattern more common in the amputed dataset.
            Must sum to reasonable proportions across all patterns.
            
        score_to_probability_func: Function mapping missingness scores to probabilities.
            - 'sigmoid-right': Higher scores → higher missing probability
            - 'sigmoid-left': Higher scores → lower missing probability  
            - 'sigmoid-mid': Medium scores → higher missing probability
            - 'sigmoid-tail': Extreme scores → higher missing probability
            - Custom Callable: User-defined mapping function
            Controls the shape of relationship between data values and
            missing probability, affecting realism of missing data patterns.
    """
    incomplete_vars: Union[List[int], List[str]]
    weights: Union[List[float], Dict[int, float], Dict[str, float]]
    mechanism: Literal['MAR', 'MCAR', 'MNAR', 'MAR+MNAR']
    freq: float
    score_to_probability_func: Union[Literal['sigmoid-right', 'sigmoid-left', 'sigmoid-mid', 'sigmoid-tail'], Callable[[List[float]], List[float]]]

class Amputation(TypedDict, total=False):
    """Configuration for introducing missing values via multivariate amputation.
    
    This TypedDict defines parameters for pyampute.ampute.MultivariateAmputation,
    which systematically introduces realistic missing value patterns into complete
    datasets following established missingness mechanisms (MCAR, MAR, MNAR).
    
    Attributes:
        prop: Overall proportion of incomplete cases (default: 0.5).
            Controls what fraction of rows will have at least one missing value.
            0.1 = 10% of rows incomplete, 0.5 = 50% incomplete.
            Higher values create more challenging imputation scenarios
            but may make some patterns unlearnable.
            
        patterns: List of missing data patterns to apply.
            Each Pattern defines a specific missingness mechanism.
            Multiple patterns can be combined to create complex,
            realistic missing data structures that test different
            aspects of imputation algorithms.
            
        std: Whether to standardize data before amputation (default: True).
            Standardization ensures all variables have equal influence
            on missing value probability calculations regardless of scale.
            Recommended for fair comparison across different feature types.
            
        verbose: Whether to print amputation progress (default: False).
            Enables detailed output about pattern application and
            missing value generation process. Useful for debugging
            complex amputation configurations.
            
        seed: Random seed for amputation process.
            Ensures reproducible missing value patterns across runs.
            Critical for consistent evaluation and debugging.
            If None, uses random seed each time.
            
        lower_range: Lower bound for binary search (default: 0.0).
            Used in iterative algorithm to achieve target proportion.
            Controls minimum threshold value in probability calculations.
            Rarely needs adjustment from default.
            
        upper_range: Upper bound for binary search (default: 10.0).
            Used in iterative algorithm to achieve target proportion.
            Controls maximum threshold value in probability calculations.
            May need adjustment for extreme proportions or custom functions.
            
        max_diff_with_target: Maximum acceptable difference from target proportion.
            Convergence criterion for iterative proportion achievement.
            Smaller values increase accuracy but may require more iterations.
            Default: 0.01 (1% tolerance).
            
        max_iter: Maximum iterations for proportion convergence (default: 100).
            Prevents infinite loops in difficult convergence cases.
            Higher values allow more precise proportion matching
            but increase computation time for complex patterns.
    """
    prop: float
    patterns: List[Pattern]
    std: bool
    verbose: bool
    seed: Optional[int]
    lower_range: float
    upper_range: float
    max_diff_with_target: float
    max_iter: int

@overload
def make_dataset(dataset_type: Literal['classification'], definition: Classification = {}, amputation: Amputation = {}, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: ...
@overload
def make_dataset(dataset_type: Literal['regression'], definition: Regression = {}, amputation: Amputation = {}, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: ...
@overload
def make_dataset(dataset_type: Literal['multilabel_classification'], definition: MultilabelClassification = {}, amputation: Amputation = {}, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

def make_dataset(
    dataset_type: Literal['classification', 'regression', 'multilabel_classification'] = "classification",
    definition: Classification | Regression | MultilabelClassification | None = None,
    amputation: Amputation | None = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Generate synthetic datasets with missing values for imputation evaluation.
    
    Args:
        dataset_type: Type of dataset to generate ('classification', 'regression', 'multilabel_classification')
        definition: Parameters for dataset generation (specific to dataset type)
        amputation: Parameters for introducing missing values
        
    Returns:
        Tuple containing:
        - X_missing: Features with missing values (no target columns)
        - X_complete: Complete features (no target columns)  
        - y: Target(s) as Series (single target) or DataFrame (multilabel)
    """
    _generate_shift_lookup_table()
    definition = definition or {}
    amputation = amputation or {}

    if seed is not None and "random_state" not in definition:
        definition["random_state"] = seed

    if seed is not None and "seed" not in amputation:
        amputation["seed"] = seed

    match dataset_type:
        case "classification":
            return _make_classification_dataset(cast(Classification, definition), amputation)
        case "regression":
            return _make_regression_dataset(cast(Regression, definition), amputation)
        case "multilabel_classification":
            return _make_multilabel_classification_dataset(cast(MultilabelClassification, definition), amputation)
        case _:
            raise ValueError("Invalid dataset type")

def _make_classification_dataset(definition: Classification, amputation: Amputation) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X, y = make_classification(**definition)
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_missing = pd.DataFrame(X_missing, columns=columns)
    df_complete = pd.DataFrame(X, columns=columns)
    y_series = pd.Series(y, name="target")

    return df_missing, df_complete, y_series

def _make_regression_dataset(definition: Regression, amputation: Amputation) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X, y = make_regression(**definition)
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_missing = pd.DataFrame(X_missing, columns=columns)
    df_complete = pd.DataFrame(X, columns=columns)
    y_series = pd.Series(y, name="target")

    return df_missing, df_complete, y_series

def _make_multilabel_classification_dataset(definition: MultilabelClassification, amputation: Amputation) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = make_multilabel_classification(**definition) # type: ignore

    if isinstance(y, spmatrix):
        y = y.todense()

    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_missing = pd.DataFrame(X_missing, columns=columns)
    df_complete = pd.DataFrame(X, columns=columns)

    target_columns = [f"target_{i}" for i in range(y.shape[1])]
    y_df = pd.DataFrame(y, columns=target_columns)

    return df_missing, df_complete, y_df

def _generate_shift_lookup_table(
    lookup_table_path: str = pyampute.utils.LOOKUP_TABLE_PATH,
    n_samples: int = int(1e6),
    lower_range: float = MultivariateAmputation.DEFAULTS["lower_range"],
    upper_range: float = MultivariateAmputation.DEFAULTS["upper_range"],
    max_iter: int = MultivariateAmputation.DEFAULTS["max_iter"],
    max_diff_with_target: float = MultivariateAmputation.DEFAULTS[
        "max_diff_with_target"
    ],
    force: bool = False,
):
    """ 
    This method was extracted from the original pyampute repository. For some reason, it's not included in the package.
    
    Args:
        lookup_table_path: Path where the lookup table will be saved
        n_samples: Number of samples to generate
        lower_range: Lower range for binary search
        upper_range: Upper range for binary search
        max_iter: Maximum number of iterations for binary search
        max_diff_with_target: Maximum difference with target for binary search
        force: If True, generate the table even if it already exists
    """
    import os
    
    # Check if lookup table already exists
    if not force and os.path.exists(lookup_table_path):
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(lookup_table_path), exist_ok=True)
    
    rng = np.random.default_rng()
    normal_sample = rng.standard_normal(size=n_samples)
    percent_missing = np.arange(0.01, 1.01, 0.01)
    score_to_prob_func_names = [
        "SIGMOID-RIGHT",
        "SIGMOID-LEFT",
        "SIGMOID-TAIL",
        "SIGMOID-MID",
    ]
    shifts = []
    for func in score_to_prob_func_names:
        shifts.append(
            [
                MultivariateAmputation._binary_search(
                    normal_sample,
                    func,
                    float(percent),
                    lower_range,
                    upper_range,
                    max_iter,
                    max_diff_with_target,
                )[0]
                for percent in percent_missing
            ]
        )
    percent_missing_2_decimal = ["{:.2f}".format(p) for p in percent_missing]
    lookup_table = pd.DataFrame(
        shifts, index=score_to_prob_func_names, columns=percent_missing_2_decimal,
    )
    lookup_table.to_csv(lookup_table_path)