import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "logs.log"

def get_logger(name: str = "ml_project") -> logging.Logger:
    logger = logging.getLogger(name)

    # Evita duplicar configuración si ya tiene handlers
    if not logger.handlers:
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")

        formatter = logging.Formatter(fmt)
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger
