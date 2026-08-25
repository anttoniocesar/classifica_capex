"""Métricas de validação, incluindo a abstenção para revisão manual."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


SECURITY_LABEL = "Cat 1 - Segurança"
MANUAL_REVIEW_LABEL = "Revisão manual"


def _safe_divide(numerator, denominator):
    """Divide contagens, usando zero para uma métrica sem denominador."""
    return float(numerator / denominator) if denominator else 0.0


def _security_metrics(y_true, y_pred):
    """Calcula a visão binária Segurança versus todas as demais saídas."""
    true_security = np.asarray(y_true, dtype=object) == SECURITY_LABEL
    predicted_security = np.asarray(y_pred, dtype=object) == SECURITY_LABEL
    tp = int(np.sum(true_security & predicted_security))
    tn = int(np.sum(~true_security & ~predicted_security))
    fp = int(np.sum(~true_security & predicted_security))
    fn = int(np.sum(true_security & ~predicted_security))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": _safe_divide(tn, tn + fp),
        "f1": _safe_divide(2 * precision * recall, precision + recall),
    }


def _classification_metrics(y_true, y_pred, metric_labels):
    """Calcula todas as métricas para um recorte da avaliação."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if len(y_true):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=metric_labels, zero_division=0
        )
        accuracy = float(accuracy_score(y_true, y_pred))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    else:
        precision = recall = f1 = np.zeros(len(metric_labels), dtype=float)
        support = np.zeros(len(metric_labels), dtype=int)
        accuracy = balanced_accuracy = 0.0

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(metric_labels)
    }
    total_support = int(support.sum())
    weights = (
        support / total_support
        if total_support
        else np.zeros_like(support, dtype=float)
    )
    return {
        "sample_count": len(y_true),
        "labels": list(metric_labels),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=metric_labels
        ),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "support": support,
        "security": _security_metrics(y_true, y_pred),
    }


def evaluate(
    y_true,
    y_pred,
    labels,
    *,
    review_mask=None,
    manual_review_label=MANUAL_REVIEW_LABEL,
):
    """Avalia o fluxo completo e, separadamente, apenas decisões automáticas.

    ``review_mask`` permite sinalizar abstenções quando ``y_pred`` ainda contém
    a classe mais próxima. Se omitida, qualquer predição igual a
    ``manual_review_label`` ou iniciada por ``"Revisão manual:"`` é reconhecida
    como revisão. No recorte completo, toda revisão é normalizada para uma única
    categoria operacional e nunca é contada como acerto.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    class_labels = list(labels)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must be one-dimensional and have equal length"
        )
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("labels must not contain duplicates")
    if manual_review_label in class_labels:
        raise ValueError("manual review label must not be one of the true class labels")
    unknown_true = sorted(set(y_true) - set(class_labels))
    if unknown_true:
        raise ValueError(f"y_true contains unknown classes: {unknown_true}")

    if review_mask is None:
        review_mask = np.asarray(
            [
                value == manual_review_label
                or (isinstance(value, str) and value.startswith("Revisão manual:"))
                for value in y_pred
            ],
            dtype=bool,
        )
    else:
        review_mask = np.asarray(review_mask, dtype=bool)
        if review_mask.ndim != 1 or len(review_mask) != len(y_true):
            raise ValueError("review_mask must be one-dimensional and match y_true")

    unknown_pred = sorted(set(y_pred[~review_mask]) - set(class_labels))
    if unknown_pred:
        raise ValueError(f"automatic predictions contain unknown classes: {unknown_pred}")
    operational_pred = y_pred.copy()
    operational_pred[review_mask] = manual_review_label
    all_metrics = _classification_metrics(
        y_true, operational_pred, class_labels + [manual_review_label]
    )
    automatic_metrics = _classification_metrics(
        y_true[~review_mask], operational_pred[~review_mask], class_labels
    )
    sample_count = len(y_true)
    review_count = int(review_mask.sum())
    review_rate = _safe_divide(review_count, sample_count)
    coverage = _safe_divide(sample_count - review_count, sample_count)

    result = {
        "all_samples": all_metrics,
        "automatic_decisions": automatic_metrics,
        "manual_review_count": review_count,
        "manual_review_rate": review_rate,
        "automatic_coverage": coverage,
    }
    # Mantém as chaves históricas como a visão conservadora de todas as amostras.
    result.update({key: value for key, value in all_metrics.items() if key != "labels"})
    return result
