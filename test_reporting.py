"""Contratos de completude e publicação atômica do relatório."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.reporting import (
    ExecutionMetadata,
    REPORT_SHEETS,
    build_report_sheets,
    export_excel,
    export_model_report,
)


def _report_data():
    table = pd.DataFrame({"project_code": ["P1"], "X01": [1.0]})
    return {
        "metadata": ExecutionMetadata(
            concept_matrix_version="1.0", dataset_version="2026.1",
            algorithm="cosseno", parameters={"eta": 0.05},
            thresholds={"similaridade": 0.7}, random_seed=42,
            record_count=1, duplicate_count=0, invalid_vector_count=0,
            executed_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
            model_version="2.0",
        ),
        "concept_matrix": pd.DataFrame({"class": ["A"], "X01": [1.0]}),
        "input_data": table,
        "similarities": [[1.0]],
        "decisions": [{"code": "P1", "review_required": True}],
        "confusion_matrix": [[1]],
        "class_metrics": {"class": "A", "f1": 1.0},
        "security_metrics": {"recall": 1.0},
        "prototype_weights": [[1.0]],
        "pca_coordinates": [[0.0, 0.0]],
        "sensitivity_analysis": {"eta": [0.05]},
        "class_names": ["A"],
    }


def test_relatorio_contem_as_doze_abas_e_metadados_obrigatorios():
    sheets = build_report_sheets(**_report_data())

    assert tuple(sheets) == REPORT_SHEETS
    assert sheets["Casos para revisão"]["code"].tolist() == ["P1"]
    fields = set(sheets["Configuração e versão"]["campo"])
    assert {"executed_at", "dataset_version", "duplicate_count", "thresholds"} <= fields


def test_exportacao_gera_excel_completo(tmp_path):
    destination = tmp_path / "relatorio.xlsx"
    export_model_report(destination, **_report_data())

    assert tuple(pd.ExcelFile(destination).sheet_names) == REPORT_SHEETS


def test_falha_de_escrita_preserva_relatorio_anterior(tmp_path, monkeypatch):
    destination = tmp_path / "relatorio.xlsx"
    destination.write_bytes(b"anterior")

    def fail(*args, **kwargs):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fail)
    with pytest.raises(RuntimeError, match="falha simulada"):
        export_excel(destination, {"aba": pd.DataFrame({"a": [1]})})

    assert destination.read_bytes() == b"anterior"
    assert list(tmp_path.iterdir()) == [destination]
