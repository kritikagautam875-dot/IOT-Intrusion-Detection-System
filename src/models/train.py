
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

SEED = 42
TARGET = "label"

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "dataset" / "train" / "train.csv"
VALIDATION_PATH = ROOT / "dataset" / "validation" / "validation.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "models"
MODEL_PATH = MODEL_DIR / "hist_gradient_boosting.joblib"


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load numeric features as float32 and preserve label strings."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Check that the CSV is in the expected dataset folder."
        )

    print(f"Loading {path} ...", flush=True)
    frame = pd.read_csv(path, dtype={TARGET: "string"})

    if TARGET not in frame.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing in {path}")

    y = frame.pop(TARGET).astype(str)
    X = frame.apply(pd.to_numeric, errors="raise").astype(np.float32)

    if X.isna().any().any():
        raise ValueError(f"Missing feature values found in {path}")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"NaN or infinite feature values found in {path}")

    return X, y


def align_features(
    X: pd.DataFrame,
    expected_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    """Require the same feature schema and order for every split."""
    missing = sorted(set(expected_columns) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected_columns))

    if missing or extra:
        raise ValueError(
            f"{split_name} feature schema mismatch. "
            f"Missing: {missing}; extra: {extra}"
        )

    return X.loc[:, expected_columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional random subset for debugging only; not a final baseline.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...", flush=True)
    X_train, y_train = load_split(TRAIN_PATH)

    if args.max_train_rows is not None:
        if args.max_train_rows < 1:
            raise ValueError("--max-train-rows must be positive")
        if args.max_train_rows < len(X_train):
            rng = np.random.RandomState(SEED)
            indices = rng.choice(
                len(X_train), size=args.max_train_rows, replace=False
            )
            indices.sort()
            X_train = X_train.iloc[indices].reset_index(drop=True)
            y_train = y_train.iloc[indices].reset_index(drop=True)
            print(
                f"DEBUG RUN: using {len(X_train):,} randomly sampled rows. "
                "Do not report these results as full-dataset results.",
                flush=True,
            )

    print("Checking constant features on training data only...", flush=True)
    constant_features = [
        col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1
    ]
    print(f"Constant features to remove: {constant_features}", flush=True)

    feature_columns = [
        col for col in X_train.columns if col not in constant_features
    ]
    X_train = X_train.loc[:, feature_columns]

    print("Loading validation data...", flush=True)
    X_val, y_val = load_split(VALIDATION_PATH)
    X_val = align_features(X_val, [*feature_columns, *constant_features], "Validation")
    X_val = X_val.loc[:, feature_columns]

    train_classes = set(y_train.unique())
    validation_classes = set(y_val.unique())
    if train_classes != validation_classes:
        raise ValueError(
            "Training and validation classes differ. "
            f"Missing from validation: {sorted(train_classes - validation_classes)}; "
            f"unseen in validation: {sorted(validation_classes - train_classes)}"
        )

    print(f"Training rows: {len(X_train):,}", flush=True)
    print(f"Validation rows: {len(X_val):,}", flush=True)
    print(f"Features retained: {len(feature_columns)}", flush=True)
    print(f"Classes: {len(train_classes)}", flush=True)
    print("Class weighting: disabled (original class distribution).", flush=True)

    # Small, explicit configuration comparison. Validation macro-F1 selects
    # the configuration; the test set is not loaded or used here.
    candidates = [
        {"max_iter": 100, "max_leaf_nodes": 15, "learning_rate": 0.1},
        {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1},
    ]

    results = []
    best_model = None
    best_score = -1.0
    best_params = None

    for index, params in enumerate(candidates, start=1):
        print(
            f"\nTraining candidate {index}/{len(candidates)}: {params}",
            flush=True,
        )
        model = HistGradientBoostingClassifier(
            **params,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-7,
            verbose=1,
        )

        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        print("Scoring on validation data...", flush=True)
        y_pred = model.predict(X_val)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        result = {
            "candidate": index,
            "parameters": params,
            "validation_macro_f1": float(macro_f1),
            "validation_weighted_f1": float(weighted_f1),
            "validation_accuracy": float(accuracy_score(y_val, y_pred)),
            "fit_seconds": round(fit_seconds, 2),
            "iterations_used": int(model.n_iter_),
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_params = params

    assert best_model is not None

    # Save a self-contained model bundle, including its exact input schema.
    bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "dropped_constant_features": constant_features,
        "target_column": TARGET,
        "classes": [str(label) for label in best_model.classes_],
        "selected_parameters": best_params,
        "selection_metric": "validation_macro_f1",
        "random_seed": SEED,
        "sklearn_version": sklearn.__version__,
        "training_rows_used": int(len(X_train)),
        "debug_subset_run": args.max_train_rows is not None,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    with (REPORT_DIR / "training_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_parameters": best_params,
                "best_validation_macro_f1": float(best_score),
                "candidates": results,
                "constant_features_removed": constant_features,
                "feature_count": len(feature_columns),
                "training_rows_used": len(X_train),
                "debug_subset_run": args.max_train_rows is not None,
            },
            f,
            indent=2,
        )

    print("\nSelected configuration:", best_params)
    print(f"Best validation macro F1: {best_score:.6f}")
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved selection report: {REPORT_DIR / 'training_selection.json'}")
    print("The test set was not accessed.", flush=True)


if __name__ == "__main__":
    main()
