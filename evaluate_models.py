import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os

print("=" * 60)
print("MODEL EVALUATION AND COMPARISON")
print("=" * 60)

# ---------------------------------------------------------
# Create output folder
# ---------------------------------------------------------

os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------------
# Load Classical NLP predictions
# ---------------------------------------------------------

print("\nLoading Classical NLP predictions...")

classical = pd.read_csv("models/classical_predictions.csv")

print("Classical columns:")
print(classical.columns.tolist())

# Detect actual/predicted columns
actual_col = None
pred_col = None

for col in classical.columns:
    if col.lower() in ["actual", "y_true", "true", "label"]:
        actual_col = col

    if col.lower() in ["predicted", "prediction", "y_pred", "pred"]:
        pred_col = col

if actual_col is None or pred_col is None:
    print("\nCould not automatically detect columns.")
    print("Please check classical_predictions.csv")
    exit()

y_true_classical = classical[actual_col]
y_pred_classical = classical[pred_col]

# ---------------------------------------------------------
# Classical Metrics
# ---------------------------------------------------------

classical_accuracy = accuracy_score(
    y_true_classical,
    y_pred_classical
)

classical_precision = precision_score(
    y_true_classical,
    y_pred_classical,
    average="weighted",
    zero_division=0
)

classical_recall = recall_score(
    y_true_classical,
    y_pred_classical,
    average="weighted",
    zero_division=0
)

classical_f1 = f1_score(
    y_true_classical,
    y_pred_classical,
    average="weighted",
    zero_division=0
)

# ---------------------------------------------------------
# Load LSTM predictions
# ---------------------------------------------------------

print("\nLoading LSTM predictions...")

lstm = pd.read_csv("models/lstm_predictions.csv")

print("LSTM columns:")
print(lstm.columns.tolist())

actual_col_lstm = None
pred_col_lstm = None

for col in lstm.columns:
    if col.lower() in ["actual", "y_true", "true", "label"]:
        actual_col_lstm = col

    if col.lower() in ["predicted", "prediction", "y_pred", "pred"]:
        pred_col_lstm = col

if actual_col_lstm is None or pred_col_lstm is None:
    print("\nCould not automatically detect LSTM columns.")
    print("Please check lstm_predictions.csv")
    exit()

y_true_lstm = lstm[actual_col_lstm]
y_pred_lstm = lstm[pred_col_lstm]

# ---------------------------------------------------------
# LSTM Metrics
# ---------------------------------------------------------

lstm_accuracy = accuracy_score(
    y_true_lstm,
    y_pred_lstm
)

lstm_precision = precision_score(
    y_true_lstm,
    y_pred_lstm,
    average="weighted",
    zero_division=0
)

lstm_recall = recall_score(
    y_true_lstm,
    y_pred_lstm,
    average="weighted",
    zero_division=0
)

lstm_f1 = f1_score(
    y_true_lstm,
    y_pred_lstm,
    average="weighted",
    zero_division=0
)

# ---------------------------------------------------------
# Print Comparison
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"\n{'Metric':<20}{'Classical NLP':<20}{'LSTM':<20}")
print("-" * 60)

print(
    f"{'Accuracy':<20}"
    f"{classical_accuracy:.4f}{'':<15}"
    f"{lstm_accuracy:.4f}"
)

print(
    f"{'Precision':<20}"
    f"{classical_precision:.4f}{'':<15}"
    f"{lstm_precision:.4f}"
)

print(
    f"{'Recall':<20}"
    f"{classical_recall:.4f}{'':<15}"
    f"{lstm_recall:.4f}"
)

print(
    f"{'F1 Score':<20}"
    f"{classical_f1:.4f}{'':<15}"
    f"{lstm_f1:.4f}"
)

# ---------------------------------------------------------
# Save comparison CSV
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Classical NLP": [
        classical_accuracy,
        classical_precision,
        classical_recall,
        classical_f1
    ],
    "LSTM": [
        lstm_accuracy,
        lstm_precision,
        lstm_recall,
        lstm_f1
    ]
})

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved:")
print("reports/model_comparison.csv")

# ---------------------------------------------------------
# Classical Confusion Matrix
# ---------------------------------------------------------

cm_classical = confusion_matrix(
    y_true_classical,
    y_pred_classical
)

print("\nClassical NLP Confusion Matrix:")
print(cm_classical)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_classical
)

disp.plot()
plt.title("Classical NLP - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "reports/classical_confusion_matrix.png",
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# LSTM Confusion Matrix
# ---------------------------------------------------------

cm_lstm = confusion_matrix(
    y_true_lstm,
    y_pred_lstm
)

print("\nLSTM Confusion Matrix:")
print(cm_lstm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lstm
)

disp.plot()
plt.title("LSTM - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "reports/lstm_confusion_matrix.png",
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# Finish
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("1. reports/model_comparison.csv")
print("2. reports/classical_confusion_matrix.png")
print("3. reports/lstm_confusion_matrix.png")