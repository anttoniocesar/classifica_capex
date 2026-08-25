import numpy as np
import pytest

from src.classifier import (
    DECISION_CLASSIFIED,
    DECISION_CLOSE_CLASSES,
    DECISION_LOW_SIMILARITY,
    calibrate_thresholds,
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
        minimum_similarity=0.0,
        minimum_margin=0.1,
    )

    np.testing.assert_array_equal(projects, original)
    np.testing.assert_allclose(scores[0], [0.6, 0.8])
    assert decisions[0].code == "P1"
    assert decisions[0].closest_class == "B"
    assert decisions[0].second_class == "A"
    assert decisions[0].margin == pytest.approx(0.2)
    assert decisions[0].status == DECISION_CLASSIFIED
    assert decisions[1].status == DECISION_CLOSE_CLASSES
    assert decisions[1].minimum_similarity == 0.0
    assert decisions[1].minimum_margin == 0.1


def test_baixa_similaridade_tem_prioridade_sobre_margem():
    _, decisions = classify(
        [[1, 1]], [[1, 0], [0, 1]], ["A", "B"],
        minimum_similarity=0.8, minimum_margin=0.2,
    )
    assert decisions[0].status == DECISION_LOW_SIMILARITY


def test_calibracao_exige_e_usa_projetos_de_todas_as_classes():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.51, 0.5]])
    thresholds = calibrate_thresholds(scores[:2], ["A", "B"], ["A", "B"])
    assert thresholds.development_sample_size == 2
    assert thresholds.development_automatic_precision == 1.0
    with pytest.raises(ValueError, match="every class"):
        calibrate_thresholds(scores[:1], ["A"], ["A", "B"])


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
