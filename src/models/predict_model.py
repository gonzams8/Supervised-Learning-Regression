"""
Realiza inferencia a partir de un CSV con características crudas:

    python src/models/predict_model.py --data path/to/new_data.csv

Imprime las predicciones al stdout y deja trazas en logs/logs.log.
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib
from src.utils.logger import get_logger   # ← import correcto

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "model.pkl"
PIPELINE_PATH = ROOT / "data" / "processed" / "preprocess.joblib"

logger = get_logger(__name__)


def predict(csv_path: str) -> None:
    logger.info("🔍 Loading inputs from %s", csv_path)
    df = pd.read_csv(csv_path)

    logger.info("📦 Loading preprocessing pipeline: %s", PIPELINE_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    logger.info("📦 Loading model: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Transform + predict
    X = pipeline.transform(df)
    preds = model.predict(X)

    logger.info("✅ Generated %d predictions", len(preds))
    for pred in preds:
        print(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Housing price prediction")
    parser.add_argument("--data", required=True, help="CSV path with raw features")
    args = parser.parse_args()
    predict(args.data)

