
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "models"
REPORTS.mkdir(parents=True, exist_ok=True)

# 1. F1-score of the 10 weakest classes
report = pd.read_csv(REPORTS / "test_classification_report.csv")
report["class"] = report["class"].astype(str)

class_report = report[
    ~report["class"].isin(
        ["accuracy", "macro avg", "weighted avg", "micro avg"]
    )
].copy()

for col in ["precision", "recall", "f1-score", "support"]:
    class_report[col] = pd.to_numeric(
        class_report[col], errors="coerce"
    )

weakest = class_report.nsmallest(10, "f1-score").sort_values("f1-score")

plt.figure(figsize=(10, 6))
plt.barh(weakest["class"], weakest["f1-score"])
plt.xlabel("F1-score")
plt.ylabel("Attack class")
plt.title("10 Weakest Classes - HistGradientBoosting Baseline")
plt.xlim(0, max(0.05, weakest["f1-score"].max() * 1.15))
plt.tight_layout()
plt.savefig(REPORTS / "weakest_classes_f1.png", dpi=300)
plt.close()


# 2. Most frequent incorrect class-to-class confusions
cm = pd.read_csv(
    REPORTS / "test_confusion_matrix.csv", index_col=0
)
cm = cm.apply(pd.to_numeric, errors="raise")

errors = []

for true_class in cm.index:
    row = cm.loc[true_class].drop(labels=true_class, errors="ignore")
    if not row.empty and row.max() > 0:
        predicted_class = row.idxmax()
        errors.append({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(row.max()),
        })

errors_df = pd.DataFrame(errors)
top_errors = errors_df.nlargest(20, "count").copy()
top_errors["pair"] = (
    top_errors["true_class"] + " → " + top_errors["predicted_class"]
)
top_errors = top_errors.sort_values("count")

plt.figure(figsize=(12, 9))
plt.barh(top_errors["pair"], top_errors["count"])
plt.xlabel("Misclassified test examples")
plt.ylabel("True class → predicted class")
plt.title("20 Most Frequent Class Confusions")
plt.tight_layout()
plt.savefig(REPORTS / "top_20_confusions.png", dpi=300)
plt.close()


# 3. Confusion-matrix heatmap for the 10 weakest classes
# Rows are true labels; columns are predicted labels.
weak_classes = weakest["class"].tolist()
labels = [label for label in cm.index if label in weak_classes]
sub_cm = cm.loc[labels, labels]

fig, ax = plt.subplots(figsize=(11, 9))
image = ax.imshow(sub_cm.values, aspect="auto")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=60, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Confusion Matrix: 10 Weakest Classes")
fig.colorbar(image, ax=ax, label="Number of examples")
plt.tight_layout()
plt.savefig(REPORTS / "weak_classes_confusion_heatmap.png", dpi=300)
plt.close()

print("Graphs saved to:", REPORTS)
print("- weakest_classes_f1.png")
print("- top_20_confusions.png")
print("- weak_classes_confusion_heatmap.png")
