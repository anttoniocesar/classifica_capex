import numpy as np
import pytest

from src.classifier import (
    DECISION_CLASSIFIED,
    DECISION_REVIEW_REQUIRED,
    calculate_margins,
    calculate_similarities,
    classify,
    get_first_and_second_classes,
    normalize_rows,
)


def test_classificador_conceitual_produz_decisao_completa_sem_alterar_entradas():
    projects = np.array([[3.0, 4.0], [1.0, 1.0]])
    concepts = np.array([[1.0, 0.0], [0.0, 1.0]])
    original = projects.copy()

    scores, decisions = classify(
        projects,
        concepts,
        ["A", "B"],
        project_codes=["P1", "P2"],
        min_margin=0.1,
    )

    np.testing.assert_array_equal(projects, original)
    np.testing.assert_allclose(scores[0], [0.6, 0.8])
    assert decisions[0].code == "P1"
    assert decisions[0].closest_class == "B"
    assert decisions[0].second_class == "A"
    assert decisions[0].margin == pytest.approx(0.2)
    assert decisions[0].status == DECISION_CLASSIFIED
    assert decisions[1].status == DECISION_REVIEW_REQUIRED


def test_funcoes_intermediarias_sao_independentes():
    normalized = normalize_rows([[3, 4]])
    np.testing.assert_allclose(normalized, [[0.6, 0.8]])
    scores = calculate_similarities([[1, 0]], [[1, 0], [0, 1]])
    first, second = get_first_and_second_classes(scores)
    np.testing.assert_array_equal(first, [0])
    np.testing.assert_array_equal(second, [1])
    np.testing.assert_allclose(calculate_margins([1.0], [0.0]), [1.0])


@pytest.mark.parametrize("argument", ([[0, 0]], [[0, 0], [1, 0]]))
def test_normalizacao_rejeita_vetor_de_norma_zero(argument):
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_rows(argument, matrix_name="project_feature_matrix")
