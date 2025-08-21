from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
from utils import get_logger  # <-- FIXED: absolute import


def _ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)


def _plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("outputs/figures/confusion_matrix.png", bbox_inches="tight")
    plt.close()


def _plot_regression(y_true, y_pred):
    # y_true vs y_pred
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig("outputs/figures/actual_vs_pred.png", bbox_inches="tight")
    plt.close()

    # residuals plot
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.savefig("outputs/figures/residuals_vs_pred.png", bbox_inches="tight")
    plt.close()


def run(config: dict):
    logger = get_logger("evaluate")
    _ensure_dirs()

    logger.info("Loading artifacts ...")
    X_train, X_test, y_train, y_test = joblib.load("data/processed/splits.joblib")
    model = joblib.load("models/preprocessor.joblib")

    problem_type = config["problem_type"]
    metrics = {}

    logger.info("Predicting on test set ...")
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision_weighted"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

        _plot_confusion(y_test, y_pred)

    else:  # regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics["rmse"] = rmse
        metrics["mae"] = mae
        metrics["r2"] = r2

        _plot_regression(y_test, y_pred)

    # Save metrics
    with open("outputs/reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Saved metrics and figures.")

    # Load the scaler and model
    scaler = joblib.load("models/preprocessor.joblib")  # Load the scaler
    model = joblib.load("models/best_model.joblib")  # Load the trained model
    # Load the test data
    X_test, y_test = joblib.load("data/processed/splits.joblib")  # Adjust as necessary
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    # Continue with evaluation logic (e.g., calculating metrics)
    # ...
