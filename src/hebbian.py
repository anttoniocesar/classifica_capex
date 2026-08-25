"""Treinamento e aplicação do protótipo Hebbiano."""
import numpy as np
from .preprocessing import normalize_vector


def unique_vectors(projects):
    """Mantém uma ocorrência de cada representação, em ordem estável."""
    values = np.asarray(projects, dtype=float)
    if values.ndim != 2:
        raise ValueError("projects deve ser uma matriz bidimensional")
    if not len(values):
        return values.copy(), np.empty(0, dtype=int)
    _, first, inverse = np.unique(values, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    return values[first[order]].copy(), np.bincount(remap[inverse])


def historical_prototype(projects):
    """Calcula ``H`` dando o mesmo peso a cada vetor distinto."""
    distinct, _ = unique_vectors(projects)
    if not len(distinct):
        raise ValueError("projects não pode ser vazio")
    return distinct.mean(axis=0)


def train_hebbian(projects, initial_prototype, eta=0.05, *, deduplicate=True):
    """Treina ``W``; por padrão cada vetor distinto contribui uma única vez."""
    weights = normalize_vector(initial_prototype)
    history = [weights.copy()]
    training = unique_vectors(projects)[0] if deduplicate else np.asarray(projects, dtype=float)
    for project in training:
        weights = normalize_vector(weights + eta * normalize_vector(project))
        history.append(weights.copy())
    return weights, np.asarray(history)


def apply_hebbian(vectors, weights):
    return np.asarray([np.dot(normalize_vector(row), normalize_vector(weights)) for row in vectors])
