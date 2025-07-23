# 🏡 California Housing - Production ML Pipeline

This project is a **production-ready regression pipeline** inspired by Chapter 2 of *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* by Aurélien Géron.

While the book provides an excellent educational walkthrough, this version is **restructured and enhanced** to follow **real-world practices** including modularization, version control, experiment tracking, testing, and reproducibility.

---

## 🚀 Objectives

- Predict housing prices using the classic California Housing dataset.
- Showcase how a simple regression problem can be scaled into a **clean, production-grade pipeline**.
- Use modern tools such as:
  - **DVC** for data and pipeline versioning.
  - **MLflow** for model tracking and experiment logging.
  - **Pytest** for testing.
  - **Scikit-learn** for modeling and preprocessing.

---

## 🧱 Project Structure

```bash
.
├── data/                   # Raw and processed data
├── models/                 # Trained models and artefacts
├── reports/                # Metrics, plots, logs
├── src/
│   ├── data/               # Data loading and splitting
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and evaluation
│   ├── utils/              # Logging and helper functions
├── tests/                  # Unit tests with pytest
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC stage versions
├── main.py                 # Entry point (run full pipeline)
├── requirements.txt
└── README.md
```

---

## 📂 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/gonzams8/california-housing-prod-pipeline.git
cd california-housing-prod-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline with DVC

```bash
dvc repro --force
```

### 5. Track experiments with MLflow

```bash
mlflow ui
# Then open http://localhost:5000 in your browser
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📌 Notes

- This pipeline can be extended to include hyperparameter tuning, deployment, and monitoring.
- The dataset is retrieved from a remote GitHub location in the current version.
- You can freely modify `src/models/train_model.py` to test different algorithms.

---

## 📘 References

- Aurélien Géron - *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- https://mlflow.org/
- https://dvc.org/
- https://scikit-learn.org/
