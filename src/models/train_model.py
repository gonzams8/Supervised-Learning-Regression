"""
Compare varios regresores mediante RandomizedSearchCV (5‑fold),
registra cada experimento en MLflow, elige el mejor modelo según
RMSE medio de validación cruzada y evalúa sobre el hold‑out.
Guarda:
    • data/models/model.pkl         (modelo entrenado)
    • data/models/metrics.json      (métricas CV + hold‑out + IC 95 %)
    • reports/model_ranking.json    (ranking completo)
El script trabaja sobre matrices transformadas que genera build_features.py.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from src.utils.logger import get_logger

# --------------------------------------------------------------------- #
# Configuración general
# --------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
SEED = 42
N_ITER = 20          # iteraciones por modelo en RandomizedSearchCV
CV_FOLDS = 5
logger = get_logger(__name__)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
# Funciones auxiliares
# --------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_train_features():
    return joblib.load(DATA_DIR / "train_features.joblib")


def load_test_features():
    return joblib.load(DATA_DIR / "test_features.joblib")


# --------------------------------------------------------------------- #
# Definición de modelos y espacios de búsqueda
# --------------------------------------------------------------------- #
MODELS = {
    "LinearRegression": (
        LinearRegression(),
        {},
    ),
    "Ridge": (
        Ridge(random_state=SEED),
        {"alpha": np.logspace(-3, 3, 50)},
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 10, 30],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 10, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=SEED),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    ),
    "XGBRegressor": (
        XGBRegressor(
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        ),
        {
            "n_estimators": [400, 800, 1200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    ),
    "LGBMRegressor": (
        LGBMRegressor(random_state=SEED, verbose=-1),   # silencioso
        {
            "n_estimators": [400, 800, 1200],
            "max_depth": [-1, 10, 30],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
        },
    ),
}

# --------------------------------------------------------------------- #
# Entrenamiento y selección
# --------------------------------------------------------------------- #
def main() -> None:
    X_train, y_train = load_train_features()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    mlflow.set_experiment("housing_price_models")
    ranking: list[dict] = []

    for name, (estimator, param_dist) in MODELS.items():
        with mlflow.start_run(run_name=name):
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=N_ITER if param_dist else 1,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=SEED,
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            search.fit(X_train, y_train)

            mean_rmse = -search.best_score_
            std_rmse = search.cv_results_["std_test_score"][search.best_index_]

            mlflow.log_metric("rmse_cv", mean_rmse)
            mlflow.log_metric("rmse_cv_std", std_rmse)
            mlflow.log_params(search.best_params_)

            ranking.append(
                {
                    "model": name,
                    "rmse_cv": mean_rmse,
                    "std": std_rmse,
                    "best_estimator": search.best_estimator_,
                    "best_params": search.best_params_,
                }
            )

            logger.info("%s | RMSE CV: %.3f (+/‑ %.3f)", name, mean_rmse, std_rmse)

    # -------------------- Ranking y mejor modelo ---------------------- #
    ranking.sort(key=lambda d: d["rmse_cv"])
    best_info = ranking[0]
    best_model_name = best_info["model"]
    best_estimator = best_info["best_estimator"]
    best_params = best_info["best_params"]

    logger.info("Best model: %s (CV RMSE %.3f)", best_model_name, best_info["rmse_cv"])

    # -------------------- Evaluación en hold‑out ---------------------- #
    X_test, y_test = load_test_features()

    feature_names = joblib.load(DATA_DIR / "preprocess.joblib").get_feature_names_out()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    y_pred = best_estimator.predict(X_test_df)

    rmse_test = rmse(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    ci = bootstrap(
        data=[(y_pred - y_test) ** 2],
        statistic=lambda s: np.sqrt(np.mean(s)),
        n_resamples=1000,
        confidence_level=0.95,
        random_state=SEED,
    )
    rmse_low, rmse_high = ci.confidence_interval

    with mlflow.start_run(run_name=f"{best_model_name}_final"):
        mlflow.log_metrics(
            {
                "rmse_test": rmse_test,
                "mae_test": mae_test,
                "r2_test": r2_test,
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
            }
        )
        mlflow.sklearn.log_model(
            best_estimator, 
            name="model",
            input_example=X_test_df.iloc[:1]
        )

    # -------------------- Persistencia ------------------------------- #
    joblib.dump(best_estimator, MODEL_DIR / "model.pkl")

    metrics_out = {
        "rmse_cv": best_info["rmse_cv"],
        "rmse_cv_std": best_info["std"],
        "rmse_test": rmse_test,
        "rmse_ci_95": [rmse_low, rmse_high],
        "mae_test": mae_test,
        "r2_test": r2_test,
        "best_model": best_model_name,
        "best_params": best_params,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=4))

    for d in ranking:
        d.pop("best_estimator", None)
    (REPORT_DIR / "model_ranking.json").write_text(json.dumps(ranking, indent=4))

    logger.info(
        "TEST RMSE: %.3f (95%% CI %.3f – %.3f) | artefacts saved to %s",
        rmse_test,
        rmse_low,
        rmse_high,
        MODEL_DIR,
    )


if __name__ == "__main__":
    main()


