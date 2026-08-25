import numpy as np
import pytest

from src.classifier import (
    ClassifierModel,
    DECISION_CLASSIFIED,
    classify_project,
    register_human_review,
)
from src.schema import CLASSES, FEATURES


def _model():
    concepts = np.zeros((len(CLASSES), len(FEATURES)))
    concepts[:, -1] = 0.1
    concepts[np.arange(len(CLASSES)), np.arange(len(CLASSES))] = 1
    return ClassifierModel(concepts, tuple(CLASSES), "modelo-2026-08")


def _features():
    values = np.zeros(len(FEATURES))
    values[0] = 1
    return values


def test_interface_retorna_resultado_auditavel_e_contribuicoes():
    result = classify_project(
        project_code="CAPEX-001",
        features=dict(zip(FEATURES, _features())),
        model=_model(),
        thresholds={"minimum_similarity": 0.8, "minimum_margin": 0.5},
    )

    assert result.code == "CAPEX-001"
    assert result.predicted_class == CLASSES[0]
    assert result.second_class == CLASSES[1]
    assert result.status == DECISION_CLASSIFIED
    assert result.review_reason is None
    assert result.model_version == "modelo-2026-08"
    assert result.contributing_features[0].feature == FEATURES[0]
    assert result.high_impact is True
    assert result.human_review is None


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ([], "vazio"),
        ([1.0] * 41, "exatamente 42"),
        ({**dict.fromkeys(FEATURES, 0.1), "X99": 1}, "desconhecidas"),
        ({index: 0.1 for index in range(42)}, "desconhecidas"),
        ([0.0] * 42, "zerados"),
    ],
)
def test_interface_rejeita_caracteristicas_invalidas(features, message):
    with pytest.raises(ValueError, match=message):
        classify_project(
            project_code="P",
            features=features,
            model=_model(),
            thresholds={"minimum_similarity": 0.5, "minimum_margin": 0.1},
        )


def test_revisao_humana_e_separada_e_identifica_responsavel():
    automatic = classify_project(
        project_code="P",
        features=_features(),
        model=_model(),
        thresholds={"minimum_similarity": 0.8, "minimum_margin": 0.5},
    )

    reviewed = register_human_review(
        automatic, reviewer="ana.silva", decision="aprovado", reason="ATA-42"
    )

    assert automatic.human_review is None
    assert reviewed.status == automatic.status
    assert reviewed.human_review.reviewer == "ana.silva"
    assert reviewed.human_review.decision == "aprovado"
    with pytest.raises(ValueError, match="revisor"):
        register_human_review(automatic, reviewer="", decision="aprovado")
