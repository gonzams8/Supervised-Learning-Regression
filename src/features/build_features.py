"""
Build the advanced feature pipeline:
- 3 ratio features       (RatioTransformer)
- log‑transformed numerics
- 10 geographic similarity features (ClusterSimilarity)
- one‑hot encoded categorical
Remainder numeric columns → imputation + scaling.
Guarda:
    • data/processed/preprocess.joblib
    • data/processed/train_features.joblib
    • data/processed/test_features.joblib
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
import joblib
from src.features.custom import ClusterSimilarity, RatioTransformer

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Helper sub‑pipelines
# --------------------------------------------------------------------------- #
def _ratio_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ratio", RatioTransformer()),
            ("scaler", StandardScaler()),
        ]
    )


log_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

default_num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# --------------------------------------------------------------------------- #
# Build full ColumnTransformer
# --------------------------------------------------------------------------- #
def _build_pipeline() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Ratio features
            ("bedrooms_ratio", _ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", _ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", _ratio_pipeline(), ["population", "households"]),
            # Log‑scaled numerics
            (
                "log_num",
                log_pipeline,
                ["total_bedrooms", "total_rooms", "population", "households", "median_income"],
            ),
            # Geographic similarity (10 features)
            (
                "geo_sim",
                ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42),
                ["latitude", "longitude"],
            ),
            # Categorical
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline,  # sólo 'housing_median_age'
        verbose_feature_names_out=True,
    )

# --------------------------------------------------------------------------- #
# Main entry‑point
# --------------------------------------------------------------------------- #
def main() -> None:
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    # Separar features y target para evitar fuga
    X_train_raw = train_df.drop(columns=["median_house_value"])
    y_train = train_df["median_house_value"].values

    X_test_raw = test_df.drop(columns=["median_house_value"])
    y_test = test_df["median_house_value"].values

    pipeline = _build_pipeline()

    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    joblib.dump(pipeline, DATA_DIR / "preprocess.joblib")
    joblib.dump((X_train, y_train), DATA_DIR / "train_features.joblib")
    joblib.dump((X_test, y_test), DATA_DIR / "test_features.joblib")
    logger.info("✅ Feature matrices and pipeline saved under %s", DATA_DIR)


if __name__ == "__main__":
    main()