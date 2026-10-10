import pandas as pd
import numpy as np


FILES = {
    "TRAIN": r"C:\UROP\IDS_FOR_IOT\dataset\train\train.csv",
    "VALIDATION": r"C:\UROP\IDS_FOR_IOT\dataset\validation\validation.csv",
    "TEST": r"C:\UROP\IDS_FOR_IOT\dataset\test\test.csv",
}


def inspect_dataset(name, filepath):
    print("\n" + "=" * 70)
    print(f"{name} DATASET")
    print("=" * 70)

    # Read only the header first
    header = pd.read_csv(filepath, nrows=0)

    print(f"Number of columns: {len(header.columns)}")
    print("\nColumns:")
    for i, column in enumerate(header.columns, start=1):
        print(f"{i:2}. {column}")

    # Read only a small sample
    df = pd.read_csv(filepath, nrows=100_000)

    print("\nData types:")
    print(df.dtypes.to_string())

    # Feature/label separation
    X = df.drop(columns=["label"])
    y = df["label"]

    # Missing values
    print("\nMissing values:")
    missing = X.isna().sum()
    print(missing[missing > 0].to_string())

    if missing.sum() == 0:
        print("No missing values found in the sample.")

    # Infinite values
    numeric_X = X.select_dtypes(include=np.number)

    inf_count = np.isinf(numeric_X).sum().sum()

    print(f"\nInfinite values: {inf_count}")

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows in sample: {duplicates}")

    # Constant features
    constant_features = X.nunique()
    constant_features = constant_features[constant_features <= 1]

    print("\nConstant features:")
    if len(constant_features) == 0:
        print("None found.")
    else:
        print(constant_features.to_string())

    # Label information
    print("\nNumber of unique labels:", y.nunique())

    print("\nLabels in sample:")
    print(y.value_counts().to_string())


def main():
    for name, filepath in FILES.items():
        inspect_dataset(name, filepath)


if __name__ == "__main__":
    main()