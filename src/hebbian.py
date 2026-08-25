"""Treinamento e aplicação do protótipo Hebbiano."""
import numpy as np
from .preprocessing import normalize_vector


def train_hebbian(projects, initial_prototype, eta=0.05):
    weights = normalize_vector(initial_prototype)
    history = [weights.copy()]
    for project in np.asarray(projects):
        weights = normalize_vector(weights + eta * normalize_vector(project))
        history.append(weights.copy())
    return weights, np.asarray(history)


def apply_hebbian(vectors, weights):
    return np.asarray([np.dot(normalize_vector(row), normalize_vector(weights)) for row in vectors])
