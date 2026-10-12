import pandas as pd

path = "reports/models/test_classification_report.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())

# Classification reports often store class names in the first column.
class_col = df.columns[0]
df[class_col] = df[class_col].astype(str)

# Exclude aggregate rows such as accuracy, macro avg and weighted avg.
df = df[~df[class_col].isin(
    ["accuracy", "macro avg", "weighted avg", "micro avg"]
)]

for col in ["precision", "recall", "f1-score", "support"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n10 classes with the lowest F1-score:")
print(
    df.sort_values("f1-score")
      [[class_col, "precision", "recall", "f1-score", "support"]]
      .head(10)
      .to_string(index=False)
)