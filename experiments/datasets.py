"""Dataset loaders for experiment evaluation.

Each loader returns (X, y) with snake_case column names and no missing values.
Columns are renamed for consistent naming across sklearn and OpenML sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

DATA_DIR = Path(__file__).parent / "data"


def _load_iris() -> tuple[pd.DataFrame, pd.Series]:
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }).dropna()
    return df.drop(columns=["target"]), df["target"]


def _load_wine() -> tuple[pd.DataFrame, pd.Series]:
    data = load_wine(as_frame=True)
    df = data.frame.rename(columns={
        "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines",
    }).dropna()
    return df.drop(columns=["target"]), df["target"]


def _load_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    data = load_diabetes(as_frame=True)
    df = data.frame.dropna()
    return df.drop(columns=["target"]), df["target"]


def _load_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={
        "mean radius": "mean_radius",
        "mean texture": "mean_texture",
        "mean perimeter": "mean_perimeter",
        "mean area": "mean_area",
        "mean smoothness": "mean_smoothness",
        "mean compactness": "mean_compactness",
        "mean concavity": "mean_concavity",
        "mean concave points": "mean_concave_points",
        "mean symmetry": "mean_symmetry",
        "mean fractal dimension": "mean_fractal_dimension",
        "radius error": "radius_error",
        "texture error": "texture_error",
        "perimeter error": "perimeter_error",
        "area error": "area_error",
        "smoothness error": "smoothness_error",
        "compactness error": "compactness_error",
        "concavity error": "concavity_error",
        "concave points error": "concave_points_error",
        "symmetry error": "symmetry_error",
        "fractal dimension error": "fractal_dimension_error",
        "worst radius": "worst_radius",
        "worst texture": "worst_texture",
        "worst perimeter": "worst_perimeter",
        "worst area": "worst_area",
        "worst smoothness": "worst_smoothness",
        "worst compactness": "worst_compactness",
        "worst concavity": "worst_concavity",
        "worst concave points": "worst_concave_points",
        "worst symmetry": "worst_symmetry",
        "worst fractal dimension": "worst_fractal_dimension",
    }).dropna()
    return df.drop(columns=["target"]), df["target"]


def _load_titanic() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_openml(data_id=40945, as_frame=True, parser="auto")
    df = data.frame.rename(columns={
        "home.dest": "home_dest",
    })
    drop_cols = ["name", "ticket", "cabin", "boat", "body", "home_dest"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()
    return df.drop(columns=["survived"]), df["survived"]


def _load_heart_disease() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df = data.frame.rename(columns={
        "chest": "chest_pain_type",
        "resting_blood_pressure": "resting_bp",
        "serum_cholestoral": "cholesterol",
        "resting_electrocardiographic_results": "resting_ecg",
        "maximum_heart_rate_achieved": "max_heart_rate",
        "exercise_induced_angina": "exercise_angina",
        "number_of_major_vessels": "num_major_vessels",
    }).dropna()
    return df.drop(columns=["class"]), df["class"]


def _load_blood_transfusion() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_openml(data_id=1464, as_frame=True, parser="auto")
    df = data.frame.rename(columns={
        "V1": "recency",
        "V2": "frequency",
        "V3": "monetary",
        "V4": "time",
    }).dropna()
    return df.drop(columns=["Class"]), df["Class"]


def _load_malaysia_house_prices() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "malaysia_house_price_data_2025.csv")
    df = df.rename(columns={
        "Township": "township",
        "Area": "area",
        "State": "state",
        "Tenure": "tenure",
        "Type": "property_type",
        "Median_Price": "median_price",
        "Median_PSF": "median_psf",
        "Transactions": "transactions",
    })
    return df.drop(columns=["median_price"]), df["median_price"]


def _load_ilpd() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_openml(data_id=1480, as_frame=True, parser="auto")
    df = data.frame.rename(columns={
        "V1": "age",
        "V2": "gender",
        "V3": "total_bilirubin",
        "V4": "direct_bilirubin",
        "V5": "alkaline_phosphotase",
        "V6": "alamine_aminotransferase",
        "V7": "aspartate_aminotransferase",
        "V8": "total_proteins",
        "V9": "albumin",
        "V10": "albumin_globulin_ratio",
    }).dropna()
    return df.drop(columns=["Class"]), df["Class"]


def _load_student_lifestyle() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "student_lifestyle_dataset.csv")
    df = df.rename(columns={
        "Student_ID": "student_id",
        "Study_Hours_Per_Day": "study_hours",
        "Extracurricular_Hours_Per_Day": "extracurricular_hours",
        "Sleep_Hours_Per_Day": "sleep_hours",
        "Social_Hours_Per_Day": "social_hours",
        "Physical_Activity_Hours_Per_Day": "physical_activity_hours",
        "GPA": "gpa",
        "Stress_Level": "stress_level",
    })
    df = df.drop(columns=["student_id"])
    return df.drop(columns=["gpa"]), df["gpa"]


DATASETS: dict[str, callable] = {
    "iris": _load_iris,
    "wine": _load_wine,
    "diabetes": _load_diabetes,
    "breast_cancer": _load_breast_cancer,
    "titanic": _load_titanic,
    "heart_disease": _load_heart_disease,
    "blood_transfusion": _load_blood_transfusion,
    "malaysia_house_prices": _load_malaysia_house_prices,
    "ilpd": _load_ilpd,
    "student_lifestyle": _load_student_lifestyle,
}


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset by name, returning (X, y) with snake_case columns."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
    return DATASETS[name]()
