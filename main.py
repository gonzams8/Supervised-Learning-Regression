"""
main.py  —  Ejecuta todo el flujo end‑to‑end sin depender de DVC.

Uso:
    python main.py                    # baja datos, procesa, entrena
    python main.py --skip-train       # omite entrenamiento
"""
import argparse
from pathlib import Path

from src.data.data_loader import download
from src.data.split_dataset import split
from src.features.build_features import main as build_feats
from src.models.train_model import main as train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_FILE = "housing.csv"
RAW_PATH = Path("data/raw") / RAW_FILE


def run_pipeline(skip_train: bool = False) -> None:
    # 1. Descargar datos
    download(RAW_FILE)

    # 2. Limpiar y hacer split estratificado
    split()

    # 3. Feature engineering
    build_feats()

    if skip_train:
        logger.info("🚫 Entrenamiento omitido (--skip-train).")
        return

    # 4. Entrenamiento y métricas
    train_model()
    logger.info("🏁 Pipeline completo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price pipeline")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Solo procesa datos y features, no entrena.",
    )
    args = parser.parse_args()
    run_pipeline(skip_train=args.skip_train)
