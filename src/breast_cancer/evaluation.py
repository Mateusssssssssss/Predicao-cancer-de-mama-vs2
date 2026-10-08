import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def evaluate_model(y_true, probabilities, threshold: float = 0.5) -> dict:
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, labels)),
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "classification_report": classification_report(
            y_true, labels, labels=[0, 1], target_names=["Benigno", "Maligno"],
            digits=3, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision_recall_curve(y_true, probabilities)[0],
        "recall_curve": precision_recall_curve(y_true, probabilities)[1],
        "predicted_labels": labels,
    }
