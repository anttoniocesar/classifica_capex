"""Métricas e matriz de confusão para a validação."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pred, labels):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)), "support": support,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)}
