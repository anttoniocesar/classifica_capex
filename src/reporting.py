"""Construção e exportação transacional do relatório auditável do modelo.

Este módulo não possui efeitos colaterais na importação.  As tabelas são
montadas integralmente em memória antes de o arquivo Excel temporário ser
aberto; o destino só é substituído depois que todas as abas foram gravadas.
"""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


REPORT_SHEETS = (
    "Configuração e versão",
    "Matriz conceitual",
    "Dados de entrada",
    "Similaridades",
    "Decisões",
    "Casos para revisão",
    "Matriz de confusão",
    "Métricas por classe",
    "Métricas Segurança",
    "Pesos dos protótipos",
    "Coordenadas PCA",
    "Análise de sensibilidade",
)


@dataclass(frozen=True)
class ExecutionMetadata:
    """Proveniência mínima que identifica uma execução reproduzível."""

    concept_matrix_version: str
    dataset_version: str
    algorithm: str
    parameters: object
    thresholds: object
    random_seed: int
    record_count: int
    duplicate_count: int
    invalid_vector_count: int
    executed_at: datetime | str | None = None
    model_version: str = "não informada"


def decisions_to_table(decisions):
    """Converte decisões, incluindo os limiares usados, para exportação."""
    result = pd.DataFrame(_record(decision) for decision in decisions)
    if result.empty and "review_required" not in result:
        result["review_required"] = pd.Series(dtype=bool)
    return result


def _record(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError("registros devem ser dicionários ou dataclasses")


def _table(value, *, columns=None):
    """Materializa uma cópia para que falhas ocorram antes da escrita."""
    if isinstance(value, pd.DataFrame):
        result = value.copy(deep=True)
    elif isinstance(value, pd.Series):
        result = value.rename(value.name or "valor").reset_index()
    elif isinstance(value, dict):
        result = pd.DataFrame([value])
    else:
        array = np.asarray(value)
        if array.ndim > 2:
            raise ValueError("uma aba deve ser uma tabela de no máximo duas dimensões")
        result = pd.DataFrame(value, columns=columns)
    # Evita uma falha tardia (e pouco clara) do openpyxl.
    if result.shape[0] > 1_048_575 or result.shape[1] > 16_384:
        raise ValueError("tabela excede os limites de uma planilha Excel")
    result = result.map(
        lambda item: _json(item) if isinstance(item, (dict, list, tuple, set)) else item
    )
    return result


def _json(value):
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def metadata_to_table(metadata):
    """Representa a configuração completa em pares ``campo``/``valor``."""
    values = asdict(metadata) if is_dataclass(metadata) else dict(metadata)
    required = {
        "concept_matrix_version", "dataset_version", "algorithm", "parameters",
        "thresholds", "random_seed", "record_count", "duplicate_count",
        "invalid_vector_count",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"metadados obrigatórios ausentes: {missing}")
    instant = values.get("executed_at") or datetime.now(timezone.utc)
    if isinstance(instant, datetime):
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        instant = instant.isoformat()
    values["executed_at"] = str(instant)
    for field in ("parameters", "thresholds"):
        values[field] = _json(values[field])
    return pd.DataFrame({"campo": list(values), "valor": list(values.values())})


def build_report_sheets(
    *, metadata, concept_matrix, input_data, similarities, decisions,
    confusion_matrix, class_metrics, security_metrics, prototype_weights,
    pca_coordinates, sensitivity_analysis, class_names=None,
):
    """Monta as doze abas obrigatórias, sem criar arquivos.

    ``decisions`` aceita dicionários ou dataclasses e alimenta também a aba de
    revisão por meio do campo booleano ``review_required``.
    """
    decision_table = decisions_to_table(decisions)
    if "review_required" not in decision_table:
        raise ValueError("decisões devem conter o campo review_required")
    review_mask = decision_table["review_required"].fillna(False).astype(bool)
    labels = list(class_names) if class_names is not None else None
    confusion = _table(confusion_matrix, columns=labels)
    if labels is not None:
        if len(confusion) != len(labels):
            raise ValueError("matriz de confusão e class_names têm dimensões diferentes")
        confusion.insert(0, "classe_real", labels)

    sheets = {
        REPORT_SHEETS[0]: metadata_to_table(metadata),
        REPORT_SHEETS[1]: _table(concept_matrix),
        REPORT_SHEETS[2]: _table(input_data),
        REPORT_SHEETS[3]: _table(similarities, columns=labels),
        REPORT_SHEETS[4]: decision_table.copy(deep=True),
        REPORT_SHEETS[5]: decision_table.loc[review_mask].reset_index(drop=True),
        REPORT_SHEETS[6]: confusion,
        REPORT_SHEETS[7]: _table(class_metrics),
        REPORT_SHEETS[8]: _table(security_metrics),
        REPORT_SHEETS[9]: _table(prototype_weights),
        REPORT_SHEETS[10]: _table(pca_coordinates),
        REPORT_SHEETS[11]: _table(sensitivity_analysis),
    }
    # Força a avaliação de todos os valores antes de iniciar qualquer I/O.
    for table in sheets.values():
        table.to_dict(orient="list")
    return sheets


def export_excel(path, sheets):
    """Grava abas atomicamente, preservando um destino anterior em caso de erro."""
    path = Path(path)
    prepared = [(str(name)[:31], _table(table)) for name, table in sheets.items()]
    names = [name for name, _ in prepared]
    if len(names) != len(set(names)):
        raise ValueError("nomes de abas duplicados após o limite de 31 caracteres")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-", suffix=path.suffix or ".xlsx",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for name, table in prepared:
                table.to_excel(writer, sheet_name=name, index=False)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def export_model_report(path, **report_data):
    """Constrói todas as abas com sucesso e somente então publica o relatório."""
    return export_excel(path, build_report_sheets(**report_data))
