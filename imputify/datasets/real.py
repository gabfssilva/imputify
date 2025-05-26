"""Real-world datasets with missing values for imputation evaluation."""

from typing import Tuple, Optional, Dict, Union, List, Callable, Literal
import os
import zipfile
from urllib.request import urlretrieve
import shutil

import numpy as np
import pandas as pd
from sklearn import datasets
from pyampute.ampute import MultivariateAmputation
from .synthetic import Amputation

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".imputify", "datasets")


def _check_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _download_file(url: str, filename: str) -> str:
    _check_cache_dir()
    filepath = os.path.join(CACHE_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        temp_file, _ = urlretrieve(url)
        
        # Validate downloaded file exists and has content
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise RuntimeError(f"Failed to download {filename}: file is empty or corrupted")
        
        shutil.move(temp_file, filepath)
    
    # Final validation of cached file
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        raise RuntimeError(f"Cached file {filename} is corrupted or missing")
    
    return filepath


def _extract_zip(zip_path: str, extract_dir: Optional[str] = None) -> str:
    if extract_dir is None:
        extract_dir = os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(zip_path))[0])
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    return extract_dir


def load_airquality(
    return_complete: bool = False,
    as_dataframe: bool = True,
    random_state: Optional[int] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the air quality dataset.

    This dataset contains hourly air quality measurements from an Italian city
    with naturally occurring missing values.

    Args:
        return_complete: Whether to return a version with imputed values.
            If True, returns the original data, imputed data, and a mask.
        as_dataframe: Whether to return pandas DataFrame or numpy array.
        random_state: Random seed for reproducibility in imputation.

    Returns:
        pd.DataFrame: Air quality dataset with missing values.
        Or if return_complete=True:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Original data, imputed data, and missing mask.
    """
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    filename = "AirQualityUCI.zip"
    
    # Download and extract dataset
    zip_path = _download_file(url, filename)
    extract_dir = _extract_zip(zip_path)
    
    # Load data
    data_path = os.path.join(extract_dir, "AirQualityUCI.csv")
    
    # This dataset uses ';' as separator and ',' as decimal separator
    data = pd.read_csv(data_path, sep=';', decimal=',')
    
    # Drop unnecessary columns and fix column names
    data = data.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1, errors='ignore')
    data.columns = [col.strip() for col in data.columns]
    
    # Replace -200 values with NaN (missing values)
    data = data.replace(-200, np.nan)
    
    # Set proper data types (all numeric for air quality measurements)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if not return_complete:
        return data
    
    # Create a copy with imputed values using median imputation
    mask = data.isna().to_numpy()
    data_imputed = data.copy()
    
    # Simple median imputation
    for col in data.columns:
        median_val = data[col].median()
        data_imputed[col] = data_imputed[col].fillna(median_val)
    
    return data, data_imputed, mask


def load_heart_disease(
    return_complete: bool = False,
    as_dataframe: bool = True,
    random_state: Optional[int] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Cleveland heart disease dataset.

    This dataset contains attributes for heart disease diagnosis with some missing values.

    Args:
        return_complete: Whether to return a version with imputed values.
            If True, returns the original data, imputed data, and a mask.
        as_dataframe: Whether to return pandas DataFrame or numpy array.
        random_state: Random seed for reproducibility in imputation.

    Returns:
        pd.DataFrame: Heart disease dataset with missing values.
        Or if return_complete=True:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Original data, imputed data, and missing mask.
    """
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    filename = "processed.cleveland.data"
    
    # Download dataset
    data_path = _download_file(url, filename)
    
    # Column names
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # Load data
    data = pd.read_csv(data_path, names=column_names, na_values='?')
    
    # Set proper data types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if not return_complete:
        return data
    
    # Create a copy with imputed values using median for numeric and mode for categorical
    mask = data.isna().to_numpy()
    data_imputed = data.copy()
    
    # Identify categorical columns
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']
    numeric_cols = [col for col in data.columns if col not in categorical_cols]
    
    # Impute missing values
    for col in categorical_cols:
        mode_val = data[col].mode()[0]
        data_imputed[col] = data_imputed[col].fillna(mode_val)
    
    for col in numeric_cols:
        median_val = data[col].median()
        data_imputed[col] = data_imputed[col].fillna(median_val)
    
    return data, data_imputed, mask


def load_breast_cancer_wisconsin(
    return_complete: bool = False,
    as_dataframe: bool = True,
    random_state: Optional[int] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Breast Cancer Wisconsin dataset.

    This dataset contains features computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass, used to predict whether the mass is malignant or benign.
    Some values are missing.

    Args:
        return_complete: Whether to return a version with imputed values.
            If True, returns the original data, imputed data, and a mask.
        as_dataframe: Whether to return pandas DataFrame or numpy array.
        random_state: Random seed for reproducibility in imputation.

    Returns:
        pd.DataFrame: Breast Cancer Wisconsin dataset with missing values.
        Or if return_complete=True:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Original data, imputed data, and missing mask.
    """
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    filename = "breast-cancer-wisconsin.data"
    
    # Download dataset
    data_path = _download_file(url, filename)
    
    # Column names
    column_names = [
        'id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape',
        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
        'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'
    ]
    
    # Load data
    data = pd.read_csv(data_path, names=column_names, na_values='?')
    
    # Drop ID column
    data = data.drop('id', axis=1)
    
    # Convert class to binary (2: benign, 4: malignant)
    data['class'] = (data['class'] == 4).astype(int)
    
    # Set proper data types
    numeric_cols = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape',
                   'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                   'bland_chromatin', 'normal_nucleoli', 'mitoses']
    categorical_cols = ['class']
    
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    
    if not return_complete:
        return data
    
    # Create a copy with imputed values
    mask = data.isna().to_numpy()
    data_imputed = data.copy()
    
    # Simple median imputation for all columns except class
    for col in data.columns:
        if col == 'class':
            # Use mode for class
            mode_val = data[col].mode()[0]
            data_imputed[col] = data_imputed[col].fillna(mode_val)
        else:
            # Use median for other columns
            median_val = data[col].median()
            data_imputed[col] = data_imputed[col].fillna(median_val)
    
    return data, data_imputed, mask


def load_titanic(
    return_complete: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Titanic dataset.

    This dataset contains passenger information and survival outcomes from the Titanic disaster,
    with naturally occurring missing values.

    Args:
        return_complete: Whether to return a version with imputed values.
            If True, returns the original data, imputed data, and a mask.
        as_dataframe: Whether to return pandas DataFrame or numpy array.
        random_state: Random seed for reproducibility in imputation.

    Returns:
        pd.DataFrame: Titanic dataset with missing values.
        Or if return_complete=True:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Original data, imputed data, and missing mask.
    """
    # Try to use seaborn's titanic dataset if available
    try:
        import seaborn as sns
        data = sns.load_dataset('titanic')
    except (ImportError, ValueError):
        # Fall back to downloading from a URL
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        filename = "titanic.csv"
        data_path = _download_file(url, filename)
        data = pd.read_csv(data_path)
    
    # Select relevant columns
    selected_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    data = data[selected_columns]
    
    # Convert categorical columns
    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data['embarked'] = data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Set proper data types
    categorical_cols = ['survived', 'pclass', 'sex', 'embarked']
    numeric_cols = ['age', 'sibsp', 'parch', 'fare']
    
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if not return_complete:
        return data
    
    # Create a copy with imputed values
    mask = data.isna().to_numpy()
    data_imputed = data.copy()
    
    # Identify categorical columns
    categorical_cols = ['pclass', 'sex', 'embarked', 'survived']
    numeric_cols = ['age', 'sibsp', 'parch', 'fare']
    
    # Impute missing values
    for col in categorical_cols:
        mode_val = data[col].mode()[0]
        data_imputed[col] = data_imputed[col].fillna(mode_val)
    
    for col in numeric_cols:
        median_val = data[col].median()
        data_imputed[col] = data_imputed[col].fillna(median_val)
    
    return data, data_imputed, mask


def load_iris(
    amputation: Amputation = {}
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Iris dataset with optional controlled amputation.

    This dataset contains 150 samples of iris flowers with 4 features each.
    The dataset is naturally complete, allowing for controlled amputation patterns.

    Args:
        amputation: Parameters for introducing missing values. If {}, returns complete data.

    Returns:
        pd.DataFrame: Iris dataset (complete or with missing values).
        Or if amputation is provided:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Data with missing values, complete data, and target variable.
    """
    data = datasets.load_iris()
    X, y = data.data, data.target
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df_complete = pd.DataFrame(X, columns=feature_names)
    
    # Set proper data types (all numeric measurements)
    for col in df_complete.columns:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    
    if not amputation:
        return df_complete
    
    # Apply amputation
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)
    df_missing = pd.DataFrame(X_missing, columns=feature_names)
    
    # Set proper data types for missing data too
    for col in df_missing.columns:
        df_missing[col] = pd.to_numeric(df_missing[col], errors='coerce')
    
    return df_missing, df_complete, y


def load_wine(
    amputation: Amputation = {}
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Wine dataset with optional controlled amputation.

    This dataset contains 178 samples of wine with 13 features each.
    The dataset is naturally complete, allowing for controlled amputation patterns.

    Args:
        amputation: Parameters for introducing missing values. If {}, returns complete data.

    Returns:
        pd.DataFrame: Wine dataset (complete or with missing values).
        Or if amputation is provided:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Data with missing values, complete data, and target variable.
    """
    data = datasets.load_wine()
    X, y = data.data, data.target
    
    df_complete = pd.DataFrame(X, columns=data.feature_names)
    
    # Set proper data types (all numeric measurements)
    for col in df_complete.columns:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    
    if not amputation:
        return df_complete
    
    # Apply amputation
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)
    df_missing = pd.DataFrame(X_missing, columns=data.feature_names)
    
    # Set proper data types for missing data too
    for col in df_missing.columns:
        df_missing[col] = pd.to_numeric(df_missing[col], errors='coerce')
    
    mask = np.isnan(X_missing)
    
    return df_missing, df_complete, y


def load_digits(
    amputation: Amputation = {}
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Digits dataset with optional controlled amputation.

    This dataset contains 1797 samples of 8x8 digit images with 64 features each.
    The dataset is naturally complete, allowing for controlled amputation patterns.

    Args:
        amputation: Parameters for introducing missing values. If {}, returns complete data.

    Returns:
        pd.DataFrame: Digits dataset (complete or with missing values).
        Or if amputation is provided:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Data with missing values, complete data, and target variable.
    """
    data = datasets.load_digits()
    X, y = data.data, data.target
    
    feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
    df_complete = pd.DataFrame(X, columns=feature_names)
    
    # Set proper data types (all numeric pixel values)
    for col in df_complete.columns:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    
    if not amputation:
        return df_complete
    
    # Apply amputation
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)
    df_missing = pd.DataFrame(X_missing, columns=feature_names)
    
    # Set proper data types for missing data too
    for col in df_missing.columns:
        df_missing[col] = pd.to_numeric(df_missing[col], errors='coerce')
    
    mask = np.isnan(X_missing)
    
    return df_missing, df_complete, y


def load_california_housing(
    amputation: Amputation = {}
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the California Housing dataset with optional controlled amputation.

    This dataset contains 20640 samples of California housing with 8 features each.
    The dataset is naturally complete, allowing for controlled amputation patterns.

    Args:
        amputation: Parameters for introducing missing values. If {}, returns complete data.

    Returns:
        pd.DataFrame: California Housing dataset (complete or with missing values).
        Or if amputation is provided:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Data with missing values, complete data, and target variable.
    """
    data = datasets.fetch_california_housing()
    X, y = data.data, data.target
    
    df_complete = pd.DataFrame(X, columns=data.feature_names)
    
    # Set proper data types (all numeric measurements)
    for col in df_complete.columns:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    
    if not amputation:
        return df_complete
    
    # Apply amputation
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)
    df_missing = pd.DataFrame(X_missing, columns=data.feature_names)
    
    # Set proper data types for missing data too
    for col in df_missing.columns:
        df_missing[col] = pd.to_numeric(df_missing[col], errors='coerce')
    
    mask = np.isnan(X_missing)
    
    return df_missing, df_complete, y


def load_diabetes(
    amputation: Amputation = {}
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load the Diabetes dataset with optional controlled amputation.

    This dataset contains 442 samples of diabetes patients with 10 features each.
    The dataset is naturally complete, allowing for controlled amputation patterns.

    Args:
        amputation: Parameters for introducing missing values. If {}, returns complete data.

    Returns:
        pd.DataFrame: Diabetes dataset (complete or with missing values).
        Or if amputation is provided:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Data with missing values, complete data, and target variable.
    """
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    
    df_complete = pd.DataFrame(X, columns=data.feature_names)
    
    # Set proper data types
    categorical_cols = ['sex']
    numeric_cols = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    
    for col in categorical_cols:
        df_complete[col] = df_complete[col].astype('category')
    for col in numeric_cols:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    
    if not amputation:
        return df_complete
    
    # Apply amputation
    ma = MultivariateAmputation(**amputation)
    X_missing = ma.fit_transform(X)
    df_missing = pd.DataFrame(X_missing, columns=data.feature_names)
    
    # Set proper data types for missing data too
    for col in categorical_cols:
        df_missing[col] = df_missing[col].astype('category')
    for col in numeric_cols:
        df_missing[col] = pd.to_numeric(df_missing[col], errors='coerce')
    
    mask = np.isnan(X_missing)
    
    return df_missing, df_complete, y