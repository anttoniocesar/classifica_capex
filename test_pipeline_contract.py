"""Contratos unitários e integração do classificador conceitual."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier import (
    DECISION_CLASSIFIED,
    DECISION_CLOSE_CLASSES,
    DECISION_LOW_SIMILARITY,
    calculate_margin,
    calculate_similarities,
    classify,
    normalize_rows,
    produce_decisions,
    rank_similarities,
)
from src.data import (
    PROJECT_METADATA,
    load_concept_matrices,
    load_project_partition,
    load_project_partitions,
)
from src.preprocessing import normalize_vector
from src.schema import CLASSES, FEATURES


def _concept_table():
    """Cria os 13 conceitos canônicos, distinguíveis e dentro da escala."""
    values = np.zeros((len(CLASSES), len(FEATURES)))
    values[np.arange(len(CLASSES)), np.arange(len(CLASSES))] = 1
    return pd.DataFrame(values, columns=FEATURES).assign(**{"class": CLASSES})[
        ["class", *FEATURES]
    ]


def _project_table(code="P-001", real_class=None, feature_index=0):
    values = np.zeros((1, len(FEATURES)))
    values[0, feature_index] = 1
    table = pd.DataFrame(values, columns=FEATURES)
    table.insert(0, "real_class", real_class or CLASSES[feature_index])
    table.insert(0, "project_code", code)
    table["classification_source"] = "base conhecida"
    table["coding_version"] = "1.0"
    table["label_validated_by"] = "teste"
    return table[["project_code", "real_class", *FEATURES, *PROJECT_METADATA[2:]]]


def _write_partitions(tmp_path, codes=("TREINO", "VALIDACAO", "TESTE")):
    paths = {}
    for name, code in zip(("train", "validation", "test"), codes):
        path = tmp_path / f"{name}.csv"
        _project_table(code=code).to_csv(path, index=False)
        paths[name] = path
    return paths


def test_normalizacao_de_vetor_nao_vazio():
    result = normalize_vector([3.0, 4.0])

    np.testing.assert_allclose(result, [0.6, 0.8])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_normalizacao_de_vetor_vazio_preserva_formato():
    result = normalize_vector([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_normalizacao_por_linhas():
    result = normalize_rows([[3.0, 4.0], [5.0, 12.0]])

    np.testing.assert_allclose(result, [[0.6, 0.8], [5 / 13, 12 / 13]])
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])


def test_matriz_conceitual_tem_dimensoes_13_por_42():
    _, _, complete = load_concept_matrices()

    assert complete.shape == (13, 42)


def test_features_correspondem_exatamente_as_colunas(tmp_path):
    table = _concept_table()
    columns = list(table.columns)
    columns[1], columns[2] = columns[2], columns[1]
    path = tmp_path / "conceitos_colunas_trocadas.csv"
    table.loc[:, columns].to_csv(path, index=False)

    with pytest.raises(ValueError, match="exatamente 42 características, nesta ordem"):
        load_concept_matrices(path)


def test_classes_correspondem_as_linhas_e_sao_reordenadas(tmp_path):
    table = _concept_table().iloc[::-1]
    path = tmp_path / "conceitos_fora_de_ordem.csv"
    table.to_csv(path, index=False)

    _, _, complete = load_concept_matrices(path)

    assert complete.shape == (len(CLASSES), len(FEATURES))
    np.testing.assert_array_equal(np.argmax(complete, axis=1), np.arange(len(CLASSES)))


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_valores_fora_da_escala_sao_rejeitados(tmp_path, invalid_value):
    table = _project_table()
    table.loc[0, FEATURES[0]] = invalid_value
    path = tmp_path / "fora_da_escala.csv"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"escala \[0.0, 1.0\]"):
        load_project_partition(path)


def test_codigos_duplicados_sao_rejeitados(tmp_path):
    table = pd.concat([_project_table(), _project_table()], ignore_index=True)
    path = tmp_path / "duplicados.csv"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="códigos de projeto duplicados"):
        load_project_partition(path)


def test_classe_desconhecida_e_rejeitada(tmp_path):
    table = _project_table(real_class="Classe inexistente")
    path = tmp_path / "classe_desconhecida.csv"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="classes reais não pertencentes a CLASSES"):
        load_project_partition(path)


def test_calculo_da_similaridade_de_cosseno():
    scores = calculate_similarities([[1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]])

    np.testing.assert_allclose(scores, [[2**-0.5, 1.0]])


def test_ordenacao_do_ranking_e_decrescente():
    ranking = rank_similarities([[0.2, 0.9, 0.5]])

    np.testing.assert_array_equal(ranking, [[1, 2, 0]])


def test_calculo_da_margem():
    assert calculate_margin(0.91, 0.73) == pytest.approx(0.18)


def test_empates_preservam_ordem_e_exigem_revisao():
    decisions = produce_decisions(
        ["P"], ["A", "B", "C"], [[0.8, 0.8, 0.1]],
        minimum_similarity=0.5,
        minimum_margin=0.01,
    )

    assert decisions[0].closest_class == "A"
    assert decisions[0].second_class == "B"
    assert decisions[0].margin == 0.0
    assert decisions[0].status == DECISION_CLOSE_CLASSES


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.70, 0.39], DECISION_CLASSIFIED),
        ([0.69, 0.20], DECISION_LOW_SIMILARITY),
        ([0.70, 0.41], DECISION_CLOSE_CLASSES),
    ],
)
def test_aplicacao_dos_limiares(scores, expected):
    decision = produce_decisions(
        ["P"], ["A", "B"], [scores],
        minimum_similarity=0.70,
        minimum_margin=0.30,
    )[0]

    assert decision.status == expected


def test_baixa_confianca_rejeita_classificacao_automatica():
    decision = produce_decisions(
        ["P"], ["A", "B"], [[0.49, 0.10]],
        minimum_similarity=0.5,
        minimum_margin=0.1,
    )[0]

    assert decision.review_required is True
    assert decision.status == DECISION_LOW_SIMILARITY


def test_particoes_sem_sobreposicao_sao_carregadas(tmp_path):
    partitions = load_project_partitions(_write_partitions(tmp_path))

    code_sets = [set(table["project_code"]) for table in partitions.values()]
    assert all(left.isdisjoint(right) for i, left in enumerate(code_sets) for right in code_sets[i + 1 :])


def test_integracao_da_carga_ate_a_decisao_final(tmp_path):
    """Uma observação idêntica ao terceiro conceito deve ser decidida sem revisão."""
    concept_path = tmp_path / "concept_matrix.csv"
    project_path = tmp_path / "projects.csv"
    _concept_table().to_csv(concept_path, index=False)
    _project_table(code="CAPEX-42", feature_index=2).to_csv(project_path, index=False)

    _, _, concepts = load_concept_matrices(concept_path)
    projects = load_project_partition(project_path)
    scores, decisions = classify(
        projects.loc[:, FEATURES].to_numpy(dtype=float),
        concepts,
        CLASSES,
        project_codes=projects["project_code"],
        minimum_similarity=0.9,
        minimum_margin=0.5,
    )

    expected_scores = np.zeros((1, len(CLASSES)))
    expected_scores[0, 2] = 1.0
    np.testing.assert_allclose(scores, expected_scores)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.code == "CAPEX-42"
    assert decision.class_index == 2
    assert decision.closest_class == CLASSES[2]
    assert decision.similarity == pytest.approx(1.0)
    assert decision.second_class == CLASSES[0]
    assert decision.second_similarity == pytest.approx(0.0)
    assert decision.margin == pytest.approx(1.0)
    assert decision.status == DECISION_CLASSIFIED
    assert decision.review_required is False
