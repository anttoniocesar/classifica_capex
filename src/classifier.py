"""Similaridade, ranking, margem e regra de decisão."""
from dataclasses import dataclass
import numpy as np
from .preprocessing import normalize_rows

@dataclass(frozen=True)
class Classification:
    class_index: int
    class_name: str
    similarity: float
    second_class: str
    second_similarity: float
    margin: float
    review_required: bool


def cosine_similarities(vectors, prototypes):
    return normalize_rows(vectors) @ normalize_rows(prototypes).T


def rank_similarities(similarities):
    return np.argsort(np.asarray(similarities), axis=1)[:, ::-1]


def classify(vectors, prototypes, classes, *, min_confidence=0.0, min_margin=0.0):
    scores = cosine_similarities(vectors, prototypes)
    ranking = rank_similarities(scores)
    results = []
    for row, (first, second, *_) in enumerate(ranking):
        margin = float(scores[row, first] - scores[row, second])
        results.append(Classification(int(first), classes[first], float(scores[row, first]),
            classes[second], float(scores[row, second]), margin,
            bool(scores[row, first] < min_confidence or margin < min_margin)))
    return scores, results
