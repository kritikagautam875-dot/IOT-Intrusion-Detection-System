
import pandas as pd

path = "reports/models/test_confusion_matrix.csv"
cm = pd.read_csv(path, index_col=0)

# Ensure the matrix contains numeric counts.
cm = cm.apply(pd.to_numeric, errors="raise")

# For each true class, find its most common incorrect prediction.
errors = []

for true_class in cm.index:
    row = cm.loc[true_class].copy()
    row = row.drop(labels=true_class, errors="ignore")

    if len(row) and row.max() > 0:
        predicted_class = row.idxmax()
        errors.append({
            "true_class": true_class,
            "most_common_wrong_prediction": predicted_class,
            "misclassified_examples": int(row.max()),
            "total_test_examples": int(cm.loc[true_class].sum()),
        })

result = pd.DataFrame(errors)

if result.empty:
    print("No incorrect predictions found in the confusion matrix.")
else:
    result = result.sort_values(
        "misclassified_examples", ascending=False
    )
    print(result.head(20).to_string(index=False))
