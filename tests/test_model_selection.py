import json, joblib, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"

def test_ranking_exists_and_sorted():
    ranking_path = ROOT / "reports" / "model_ranking.json"
    assert ranking_path.exists()
    ranking = json.loads(ranking_path.read_text())
    rmses = [r["rmse_cv"] for r in ranking]
    assert rmses == sorted(rmses)

def test_model_can_predict():
    model = joblib.load(MODEL_DIR / "model.pkl")
    sample = np.random.rand(5, model.n_features_in_)
    preds = model.predict(sample)
    assert preds.shape == (5,)
