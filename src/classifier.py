"""Classificador conceitual por similaridade de cosseno.

Todas as funções deste módulo são puras: elas não treinam pesos, não alteram as
matrizes recebidas e não escrevem resultados em disco.
"""

from dataclasses import dataclass

import numpy as np


DECISION_CLASSIFIED = "Classificação automática"
DECISION_LOW_SIMILARITY = "Revisão manual: baixa similaridade"
DECISION_CLOSE_CLASSES = "Revisão manual: classes próximas"
DECISION_REVIEW_REQUIRED = "review_required"  # compatibilidade para consumidores antigos


@dataclass(frozen=True)
class DecisionThresholds:
    """Limiares calibrados em uma partição de desenvolvimento identificada."""

    minimum_similarity: float
    minimum_margin: float
    development_sample_size: int
    development_automatic_precision: float
    development_automatic_coverage: float


@dataclass(frozen=True)
class Classification:
    """Decisão auditável para um projeto."""

    code: str
    class_index: int
    closest_class: str
    similarity: float
    second_class: str
    second_similarity: float
    margin: float
    status: str
    minimum_similarity: float
    minimum_margin: float

    @property
    def class_name(self):
        """Nome compatível com a interface anterior."""
        return self.closest_class

    @property
    def review_required(self):
        """Indica se os limiares encaminharam a decisão para revisão."""
        return self.status != DECISION_CLASSIFIED


def calibrate_thresholds(
    similarity_matrix,
    true_classes,
    class_names,
    *,
    target_automatic_precision=1.0,
):
    """Calibra limiares maximizando cobertura na base de desenvolvimento.

    Somente pares que atingem ``target_automatic_precision`` entre as decisões
    automáticas são elegíveis. A cobertura desempata em favor dos menores
    limiares. Exigir todas as classes evita calibrar a rejeição com um recorte
    de uma única categoria (por exemplo, os 19 projetos de Segurança).
    """
    scores = np.asarray(similarity_matrix, dtype=float)
    classes = list(class_names)
    labels = list(true_classes)
    if scores.ndim != 2 or scores.shape != (len(labels), len(classes)):
        raise ValueError(
            "development similarities, labels and classes have incompatible shapes"
        )
    if not 0 < target_automatic_precision <= 1:
        raise ValueError("target_automatic_precision must be in (0, 1]")
    missing = sorted(set(classes) - set(labels))
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"development data contains unknown classes: {unknown}")
    if missing:
        raise ValueError(
            "development data must contain projects from every class; "
            f"missing: {missing}"
        )

    first, second = get_first_and_second_classes(scores)
    rows = np.arange(len(labels))
    winners = scores[rows, first]
    margins = winners - scores[rows, second]
    correct = np.asarray(
        [classes[index] == label for index, label in zip(first, labels)]
    )
    sim_candidates = np.unique(np.append(winners, np.nextafter(winners.max(), np.inf)))
    margin_candidates = np.unique(np.append(margins, np.nextafter(margins.max(), np.inf)))
    best = None
    for minimum_similarity in sim_candidates:
        for minimum_margin in margin_candidates:
            automatic = (winners >= minimum_similarity) & (margins >= minimum_margin)
            count = int(automatic.sum())
            if not count:
                continue
            precision = float(correct[automatic].mean())
            if precision < target_automatic_precision:
                continue
            candidate = (count, -minimum_similarity, -minimum_margin, precision)
            if best is None or candidate > best[0]:
                best = (candidate, minimum_similarity, minimum_margin, precision)
    if best is None:
        raise ValueError("no thresholds meet target_automatic_precision on development data")
    _, minimum_similarity, minimum_margin, precision = best
    return DecisionThresholds(
        minimum_similarity=float(minimum_similarity),
        minimum_margin=float(minimum_margin),
        development_sample_size=len(labels),
        development_automatic_precision=precision,
        development_automatic_coverage=best[0][0] / len(labels),
    )


def normalize_rows(matrix, *, matrix_name="matrix"):
    """Retorna uma cópia normalizada por linha e rejeita vetores de norma zero."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{matrix_name} must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{matrix_name} contains non-finite values")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] == 0).tolist()
    if zero_rows:
        raise ValueError(f"{matrix_name} contains zero-norm vectors at rows {zero_rows}")
    return values / norms


def calculate_similarities(project_feature_matrix, class_concept_matrix):
    """Calcula cossenos projeto × classe a partir das matrizes explicitadas."""
    projects = normalize_rows(
        project_feature_matrix, matrix_name="project_feature_matrix"
    )
    concepts = normalize_rows(
        class_concept_matrix, matrix_name="class_concept_matrix"
    )
    if projects.shape[1] != concepts.shape[1]:
        raise ValueError(
            "project_feature_matrix and class_concept_matrix must have the "
            "same number of columns"
        )
    return projects @ concepts.T


def cosine_similarities(project_feature_matrix, class_concept_matrix):
    """Nome legado para :func:`calculate_similarities`."""
    return calculate_similarities(project_feature_matrix, class_concept_matrix)


def rank_similarities(similarity_matrix):
    """Ordena classes por similaridade, preservando a ordem em caso de empate."""
    scores = np.asarray(similarity_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError("similarity_matrix must be a two-dimensional matrix")
    if scores.shape[1] < 2:
        raise ValueError("similarity_matrix must contain at least two classes")
    if not np.isfinite(scores).all():
        raise ValueError("similarity_matrix contains non-finite values")
    return np.argsort(-scores, axis=1, kind="stable")


def get_first_and_second_classes(similarity_matrix):
    """Retorna os índices da primeira e da segunda classe de cada projeto."""
    ranking = rank_similarities(similarity_matrix)
    return ranking[:, 0], ranking[:, 1]


def calculate_margins(first_similarities, second_similarities):
    """Calcula a diferença entre as similaridades da primeira e segunda classes."""
    first = np.asarray(first_similarities, dtype=float)
    second = np.asarray(second_similarities, dtype=float)
    if first.shape != second.shape:
        raise ValueError("first and second similarities must have the same shape")
    return first - second


def calculate_margin(first_similarity, second_similarity):
    """Calcula uma margem escalar (atalho para decisões unitárias)."""
    margin = calculate_margins(first_similarity, second_similarity)
    return float(margin) if margin.ndim == 0 else margin


def produce_decisions(
    project_codes,
    class_names,
    similarity_matrix,
    *,
    minimum_similarity,
    minimum_margin,
):
    """Produz uma decisão estruturada para cada linha de ``similarity_matrix``."""
    scores = np.asarray(similarity_matrix, dtype=float)
    first_indices, second_indices = get_first_and_second_classes(scores)
    codes = list(project_codes)
    classes = list(class_names)
    if len(codes) != scores.shape[0]:
        raise ValueError("project_codes must identify every project matrix row")
    if len(classes) != scores.shape[1]:
        raise ValueError("class_names must identify every concept matrix row")

    rows = np.arange(scores.shape[0])
    first_scores = scores[rows, first_indices]
    second_scores = scores[rows, second_indices]
    margins = calculate_margins(first_scores, second_scores)
    decisions = []
    for row, code in enumerate(codes):
        if first_scores[row] < minimum_similarity:
            status = DECISION_LOW_SIMILARITY
        elif margins[row] < minimum_margin:
            status = DECISION_CLOSE_CLASSES
        else:
            status = DECISION_CLASSIFIED
        decisions.append(
            Classification(
                code=str(code),
                class_index=int(first_indices[row]),
                closest_class=classes[first_indices[row]],
                similarity=float(first_scores[row]),
                second_class=classes[second_indices[row]],
                second_similarity=float(second_scores[row]),
                margin=float(margins[row]),
                status=status,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
            )
        )
    return decisions


def classify(
    project_feature_matrix,
    class_concept_matrix,
    class_names,
    *,
    project_codes=None,
    minimum_similarity,
    minimum_margin,
):
    """Executa o baseline conceitual, sem treinamento ou pesos Hebbianos."""
    scores = calculate_similarities(project_feature_matrix, class_concept_matrix)
    if project_codes is None:
        project_codes = [str(index) for index in range(scores.shape[0])]
    decisions = produce_decisions(
        project_codes,
        class_names,
        scores,
        minimum_similarity=minimum_similarity,
        minimum_margin=minimum_margin,
    )
    return scores, decisions
