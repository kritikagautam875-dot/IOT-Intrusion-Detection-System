
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Update DATASET_ROOT if your CSVs are stored elsewhere.
DATASET_ROOT = PROJECT_ROOT / "dataset"

TRAIN_PATH = DATASET_ROOT / "train" / "train.csv"
VALIDATION_PATH = DATASET_ROOT / "validation" / "validation.csv"
TEST_PATH = DATASET_ROOT / "test" / "test.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

LABEL_COLUMN = "label"


# HELPER FUNCTIONS


def check_files():
    """Verify that all dataset files exist."""
    paths = {
        "Train": TRAIN_PATH,
        "Validation": VALIDATION_PATH,
        "Test": TEST_PATH,
    }

    print("\nChecking dataset paths...")

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found: {path}\n"
                "Check DATASET_ROOT in eda.py."
            )

        print(f"{name}: {path}")


def get_dataset_info(path):
    """Collect dataset metadata without loading the entire CSV."""
    total_rows = 0
    label_counts = {}
    missing_counts = None
    infinite_counts = None
    feature_min = None
    feature_max = None
    numeric_columns = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"Missing target column: {LABEL_COLUMN}")

        X = chunk.drop(columns=[LABEL_COLUMN])
        numeric = X.select_dtypes(include=np.number)

        if numeric.shape[1] != X.shape[1]:
            raise ValueError("One or more feature columns are non-numeric.")

        if numeric_columns is None:
            numeric_columns = list(numeric.columns)

        missing = X.isna().sum()
        inf = pd.Series(
            np.isinf(numeric.to_numpy()).sum(axis=0),
            index=numeric.columns,
        )

        missing_counts = (
            missing if missing_counts is None
            else missing_counts.add(missing, fill_value=0)
        )
        infinite_counts = (
            inf if infinite_counts is None
            else infinite_counts.add(inf, fill_value=0)
        )

        mins = numeric.min()
        maxs = numeric.max()

        feature_min = (
            mins if feature_min is None
            else pd.concat([feature_min, mins], axis=1).min(axis=1)
        )
        feature_max = (
            maxs if feature_max is None
            else pd.concat([feature_max, maxs], axis=1).max(axis=1)
        )

        counts = chunk[LABEL_COLUMN].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

    return {
        "rows": total_rows,
        "features": numeric_columns,
        "labels": label_counts,
        "missing": missing_counts,
        "infinite": infinite_counts,
        "minimum": feature_min,
        "maximum": feature_max,
    }


def save_label_distribution(label_counts, split_name):
    """Save class counts and a class-distribution plot."""
    counts = pd.Series(label_counts, name="count")
    counts = counts.sort_values(ascending=False)

    counts.to_csv(REPORT_DIR / f"{split_name}_class_counts.csv")

    plt.figure(figsize=(14, 9))
    counts.sort_values().plot(kind="barh")

    plt.title(f"{split_name.title()} - Attack Class Distribution")
    plt.xlabel("Number of records")
    plt.ylabel("Label")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{split_name}_class_distribution.png",
        dpi=200,
    )
    plt.close()

    print(f"\n{split_name.title()} class distribution saved.")



def get_random_sample(path, sample_size=100_000):
    """Create a bounded random sample efficiently using chunks."""
    import pandas as pd

    chunks = []
    rows_seen = 0

    for chunk in pd.read_csv(path, chunksize=100_000):
        rows_seen += len(chunk)

        # Take a small random sample from each chunk
        remaining = sample_size - sum(len(c) for c in chunks)
        if remaining <= 0:
            break

        take = min(len(chunk), max(1, remaining))
        chunks.append(chunk.sample(n=take, random_state=42 + rows_seen))

    sample = pd.concat(chunks, ignore_index=True)

    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=42)

    return sample



def analyze_sample(sample):
    """Analyze distributions and correlations on a bounded sample."""
    X = sample.drop(columns=[LABEL_COLUMN])
    y = sample[LABEL_COLUMN]

    # Descriptive statistics
    X.describe().T.to_csv(REPORT_DIR / "feature_statistics.csv")

    # Label distribution in the sample
    sample_counts = y.value_counts()
    sample_counts.to_csv(REPORT_DIR / "sample_class_counts.csv")

    # Correlation analysis
    corr = X.corr()

    corr.to_csv(REPORT_DIR / "feature_correlations.csv")

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "correlation_heatmap.png",
        dpi=200,
    )
    plt.close()

    # Distribution plots for selected features
    selected_features = [
        "flow_duration",
        "Header_Length",
        "Rate",
        "Srate",
        "IAT",
        "Covariance",
        "Variance",
        "Weight",
    ]

    selected_features = [
        feature for feature in selected_features
        if feature in X.columns
    ]

    for feature in selected_features:
        plt.figure(figsize=(9, 5))
        sns.histplot(X[feature], bins=50, kde=False)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"distribution_{feature}.png",
            dpi=150,
        )
        plt.close()

    # Boxplots on a sample; extreme values may dominate the display.
    box_features = [
        "flow_duration",
        "Rate",
        "IAT",
        "Covariance",
        "Variance",
    ]

    box_features = [
        feature for feature in box_features
        if feature in X.columns
    ]

    for feature in box_features:
        plt.figure(figsize=(9, 4))
        sns.boxplot(x=X[feature])
        plt.title(f"Outlier overview: {feature}")
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"boxplot_{feature}.png",
            dpi=150,
        )
        plt.close()

    print("\nSample analysis complete.")



# MAIN


def main():
    check_files()

    split_paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
    }

    all_info = {}

    for split_name, path in split_paths.items():
        print("\n" + "=" * 65)
        print(f"ANALYZING {split_name.upper()}")
        print("=" * 65)

        info = get_dataset_info(path)
        all_info[split_name] = info

        print(f"Rows: {info['rows']:,}")
        print(f"Features: {len(info['features'])}")
        print(f"Classes: {len(info['labels'])}")
        print(f"Missing feature values: {int(info['missing'].sum())}")
        print(f"Infinite feature values: {int(info['infinite'].sum())}")

        constant_features = info["minimum"][
            info["minimum"] == info["maximum"]
        ]

        print("\nConstant features:")
        print(
            constant_features.index.tolist()
            if len(constant_features)
            else "None"
        )

        save_label_distribution(info["labels"], split_name)

    # Verify schema consistency across splits.
    train_features = all_info["train"]["features"]
    for split_name in ("validation", "test"):
        if all_info[split_name]["features"] != train_features:
            raise ValueError(
                f"Feature schema mismatch: train vs {split_name}"
            )

    print("\nFeature schema matches across all splits.")

    # Perform detailed distribution/correlation analysis on TRAIN only.
    # Validation and test remain untouched for model evaluation.
    print("\nCreating a bounded random sample from training data...")
    sample = get_random_sample(TRAIN_PATH)

    sample.to_csv(
        REPORT_DIR / "train_eda_sample.csv",
        index=False,
    )

    analyze_sample(sample)

    # Save a concise audit summary.
    summary = pd.DataFrame({
        split: {
            "rows": info["rows"],
            "features": len(info["features"]),
            "classes": len(info["labels"]),
            "missing_values": int(info["missing"].sum()),
            "infinite_values": int(info["infinite"].sum()),
        }
        for split, info in all_info.items()
    }).T

    summary.to_csv(REPORT_DIR / "dataset_summary.csv")

    print("\n" + "=" * 65)
    print("EDA COMPLETED")
    print("=" * 65)
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
