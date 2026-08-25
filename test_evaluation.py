import numpy as np

from src.evaluation import MANUAL_REVIEW_LABEL, evaluate


LABELS = ["Cat 1 - Segurança", "Cat 2", "Cat 3"]


def test_evaluate_reporta_fluxo_completo_e_decisoes_automaticas():
    result = evaluate(
        ["Cat 1 - Segurança", "Cat 1 - Segurança", "Cat 2", "Cat 3", "Cat 2"],
        ["Cat 1 - Segurança", MANUAL_REVIEW_LABEL, "Cat 1 - Segurança", "Cat 3", "Cat 3"],
        LABELS,
    )

    assert result["manual_review_count"] == 1
    assert result["manual_review_rate"] == 0.2
    assert result["automatic_coverage"] == 0.8
    assert result["all_samples"]["labels"] == LABELS + [MANUAL_REVIEW_LABEL]
    assert result["all_samples"]["confusion_matrix"].shape == (4, 4)
    assert result["automatic_decisions"]["confusion_matrix"].shape == (3, 3)
    assert result["all_samples"]["accuracy"] == 0.4
    assert result["automatic_decisions"]["accuracy"] == 0.5
    assert result["all_samples"]["per_class"]["Cat 2"]["recall"] == 0.0
    assert "weighted_f1" in result["automatic_decisions"]


def test_visao_seguranca_e_conservadora_quando_ha_revisao():
    result = evaluate(
        ["Cat 1 - Segurança", "Cat 1 - Segurança", "Cat 2", "Cat 3"],
        ["Cat 1 - Segurança", "Cat 2", "Cat 1 - Segurança", "Cat 3"],
        LABELS,
        review_mask=[False, True, False, False],
    )

    security = result["all_samples"]["security"]
    assert security == {
        "true_positives": 1,
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "sensitivity": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
    }
    assert result["automatic_decisions"]["security"]["false_negatives"] == 0


def test_evaluate_trata_recorte_automatico_vazio():
    result = evaluate(["Cat 2"], [MANUAL_REVIEW_LABEL], LABELS)
    assert result["automatic_coverage"] == 0.0
    assert result["automatic_decisions"]["sample_count"] == 0
    np.testing.assert_array_equal(
        result["automatic_decisions"]["confusion_matrix"], np.zeros((3, 3), dtype=int)
    )
