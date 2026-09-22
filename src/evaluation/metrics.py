"""Métricas de evaluación para el problema de clasificación binaria (Churn)."""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="Yes"),
        "recall": recall_score(y_true, y_pred, pos_label="Yes"),
        "f1": f1_score(y_true, y_pred, pos_label="Yes"),
        "roc_auc": roc_auc_score((y_true == "Yes").astype(int), y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
