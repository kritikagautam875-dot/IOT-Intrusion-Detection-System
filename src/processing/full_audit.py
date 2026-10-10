import pandas as pd
import numpy as np

TRAIN_PATH = r"C:\UROP\IDS_FOR_IOT\dataset\train\train.csv"

CHUNK_SIZE = 100_000


def main():
    print("=" * 70)
    print("FULL TRAINING DATA AUDIT")
    print("=" * 70)

    total_rows = 0
    total_duplicates = 0

    missing_counts = None
    infinite_counts = None

    min_values = None
    max_values = None

    feature_unique_values = None

    label_counts = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        print(f"\rProcessing chunk {chunk_number}...", end="")

        total_rows += len(chunk)

        # Separate features and label
        X = chunk.drop(columns=["label"])

        # Missing values
        chunk_missing = X.isna().sum()

        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts += chunk_missing

        # Infinite values
        numeric_X = X.select_dtypes(include=np.number)
        chunk_inf = np.isinf(numeric_X).sum()

        if infinite_counts is None:
            infinite_counts = chunk_inf
        else:
            infinite_counts += chunk_inf

        # Duplicate rows within each chunk
        total_duplicates += chunk.duplicated().sum()

        # Min / Max
        chunk_min = numeric_X.min()
        chunk_max = numeric_X.max()

        if min_values is None:
            min_values = chunk_min
            max_values = chunk_max
        else:
            min_values = pd.concat([min_values, chunk_min], axis=1).min(axis=1)
            max_values = pd.concat([max_values, chunk_max], axis=1).max(axis=1)

        # Unique values
        chunk_unique = X.nunique()

        if feature_unique_values is None:
            feature_unique_values = chunk_unique
        else:
            # We only need to know whether a feature has >1 value.
            feature_unique_values = pd.concat(
                [feature_unique_values, chunk_unique],
                axis=1
            ).max(axis=1)

        # Labels
        counts = chunk["label"].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + count

    print("\n\n" + "=" * 70)
    print("AUDIT RESULTS")
    print("=" * 70)

    print(f"\nTotal rows: {total_rows:,}")

    print(f"\nDuplicate rows within chunks: {total_duplicates:,}")

    print("\nMissing values:")
    missing = missing_counts[missing_counts > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    print("\nInfinite values:")

    infinite = infinite_counts[infinite_counts > 0]

    if len(infinite) == 0:
        print("None")
    else:
        print(infinite)

    print("\nConstant features:")

    constant = feature_unique_values[feature_unique_values <= 1]

    if len(constant) == 0:
        print("None")
    else:
        print(constant)

    print("\nMinimum values:")
    print(min_values.to_string())

    print("\nMaximum values:")
    print(max_values.to_string())

    print("\nLabel distribution:")
    label_series = pd.Series(label_counts).sort_values(ascending=False)
    print(label_series.to_string())

    print("\nNumber of classes:", len(label_series))

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()