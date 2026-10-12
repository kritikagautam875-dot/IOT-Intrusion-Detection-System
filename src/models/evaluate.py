
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TARGET = "label"
ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "dataset" / "test" / "test.csv"
MODEL_PATH = ROOT / "models" / "hist_gradient_boosting.joblib"
REPORT_DIR = ROOT / "reports" / "models"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun python src/models/train.py first."
        )
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if bundle.get("debug_subset_run", False):
        raise RuntimeError(
            "The saved model was trained on a debug subset. "
            "Train the full baseline before final test evaluation."
        )

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    classes = [str(label) for label in bundle["classes"]]

    print(f"Loading held-out test data: {TEST_PATH}", flush=True)
    test_frame = pd.read_csv(TEST_PATH, dtype={TARGET: "string"})

    if TARGET not in test_frame.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")

    y_test = test_frame.pop(TARGET).astype(str)
    missing = sorted(set(feature_columns) - set(test_frame.columns))
    if missing:
        raise ValueError(f"Test data is missing required features: {missing}")

    X_test = test_frame.loc[:, feature_columns].apply(
        pd.to_numeric, errors="raise"
    ).astype(np.float32)

    if X_test.isna().any().any():
        raise ValueError("Missing feature values found in the test data.")
    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("NaN or infinite feature values found in the test data.")

    unseen_labels = sorted(set(y_test.unique()) - set(classes))
    if unseen_labels:
        raise ValueError(f"Test data contains labels not seen in training: {unseen_labels}")

    print(f"Test rows: {len(X_test):,}", flush=True)
    print(f"Features: {len(feature_columns)}", flush=True)
    print("Generating predictions...", flush=True)

    started = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_seconds = time.perf_counter() - started

    metrics = {
        "dataset": "CICIoT2023 test split",
        "test_rows": int(len(y_test)),
        "feature_count": int(len(feature_columns)),
        "class_count_in_model": int(len(classes)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "prediction_seconds": round(prediction_seconds, 2),
        "sklearn_version": sklearn.__version__,
        "selected_parameters": bundle.get("selected_parameters"),
        "training_rows_used": bundle.get("training_rows_used"),
        "random_seed": bundle.get("random_seed"),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Explicit class ordering makes the matrix consistent and interpretable.
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(classes, name="actual"),
        columns=pd.Index(classes, name="predicted"),
    )
    matrix_df.to_csv(REPORT_DIR / "test_confusion_matrix.csv")

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "test_classification_report.csv",
        index_label="class",
    )

    with (REPORT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFINAL TEST METRICS")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics: {REPORT_DIR / 'test_metrics.json'}")
    print(f"Saved per-class report: {REPORT_DIR / 'test_classification_report.csv'}")
    print(f"Saved confusion matrix: {REPORT_DIR / 'test_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
