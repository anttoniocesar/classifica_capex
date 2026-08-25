"""Normalização e validação numérica dos vetores."""
import numpy as np


def validate_vectors(matrix, *, expected_columns=None, name="matriz"):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} deve ser bidimensional")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(f"{name} deve possuir {expected_columns} colunas")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contém valores não finitos")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError(f"{name} contém vetor nulo")
    return values


def normalize_rows(matrix):
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def normalize_vector(vector):
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value)
    return value.copy() if norm == 0 else value / norm
