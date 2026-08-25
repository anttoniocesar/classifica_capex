"""Carregamento e validação dos dados versionados do classificador."""
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CLASSES, FEATURES, validate_feature_values


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONCEPT_MATRIX_PATH = DATA_DIR / "concept_matrix.csv"
PROJECT_PARTITIONS = {
    "train": DATA_DIR / "projects_train.csv",
    "validation": DATA_DIR / "projects_validation.csv",
    "test": DATA_DIR / "projects_test.csv",
}
PROJECT_METADATA = (
    "project_code",
    "real_class",
    "classification_source",
    "coding_version",
    "label_validated_by",
)


def project_feature_fingerprint(table):
    """Retorna a assinatura dos 42 atributos para auditar cópias/variantes.

    A assinatura não é usada como identidade do projeto: projetos distintos podem
    legitimamente receber a mesma codificação. Ela serve para produzir candidatos
    à revisão humana antes de qualquer divisão da amostra.
    """
    import hashlib

    values = _numeric_features(table, "projetos para assinatura")
    return [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in values]


def find_cross_partition_feature_duplicates(partitions):
    """Lista vetores idênticos que foram colocados em partições diferentes."""
    occurrences = {}
    for partition, table in partitions.items():
        for code, fingerprint in zip(table["project_code"], project_feature_fingerprint(table)):
            occurrences.setdefault(fingerprint, []).append((partition, code))
    return {
        fingerprint: projects
        for fingerprint, projects in occurrences.items()
        if len({partition for partition, _ in projects}) > 1
    }


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"arquivo de dados não encontrado: {path}")
    return pd.read_csv(path, keep_default_na=False)


def _numeric_features(table, context):
    """Converte somente as 42 colunas canônicas e valida seu domínio."""
    try:
        values = table.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: todas as características devem ser numéricas") from exc
    validate_feature_values(values, context)
    return values


def load_concept_matrices(path=CONCEPT_MATRIX_PATH):
    """Lê a matriz conceitual CSV e retorna ``(X01:X32, X33:X42, completa)``."""
    table = _read_csv(path)
    expected = ["class", *FEATURES]
    if list(table.columns) != expected:
        raise ValueError(
            "matriz conceitual deve conter uma coluna 'class' e exatamente "
            f"{len(FEATURES)} características, nesta ordem"
        )
    if table["class"].duplicated().any():
        raise ValueError("matriz conceitual contém classes duplicadas")
    if set(table["class"]) != set(CLASSES) or len(table) != len(CLASSES):
        raise ValueError("matriz conceitual deve conter exatamente as classes de CLASSES")
    table = table.set_index("class").loc[CLASSES].reset_index()
    concepts = _numeric_features(table, "matriz conceitual")
    return concepts[:, :32].copy(), concepts[:, 32:].copy(), concepts


def load_project_partition(path):
    """Lê uma partição, valida esquema, metadados e valores, e devolve a tabela."""
    table = _read_csv(path)
    expected = ["project_code", "real_class", *FEATURES, *PROJECT_METADATA[2:]]
    if list(table.columns) != expected:
        raise ValueError(
            "partição deve conter código, classe real, exatamente "
            f"{len(FEATURES)} características e os três campos de proveniência"
        )
    if table["project_code"].duplicated().any():
        duplicates = table.loc[table["project_code"].duplicated(), "project_code"].tolist()
        raise ValueError(f"códigos de projeto duplicados: {duplicates}")
    for column in PROJECT_METADATA:
        if table[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"campo obrigatório vazio: {column}")
    invalid_classes = sorted(set(table["real_class"]) - set(CLASSES))
    if invalid_classes:
        raise ValueError(f"classes reais não pertencentes a CLASSES: {invalid_classes}")
    values = _numeric_features(table, f"partição {Path(path).name}")
    result = table.copy()
    result.loc[:, FEATURES] = values
    return result


def load_project_partitions(paths=None):
    """Carrega as três partições e impede vazamento do mesmo código entre elas."""
    paths = PROJECT_PARTITIONS if paths is None else paths
    missing = set(PROJECT_PARTITIONS) - set(paths)
    if missing:
        raise ValueError(f"partições obrigatórias ausentes: {sorted(missing)}")
    partitions = {name: load_project_partition(paths[name]) for name in PROJECT_PARTITIONS}
    occurrences = {}
    for name, table in partitions.items():
        for code in table["project_code"]:
            occurrences.setdefault(code, []).append(name)
    overlap = {code: names for code, names in occurrences.items() if len(names) > 1}
    if overlap:
        raise ValueError(f"projetos presentes em mais de uma partição: {overlap}")
    return partitions


def load_projects(path=PROJECT_PARTITIONS["train"]):
    """Interface compatível: retorna códigos e matriz numérica de uma partição CSV."""
    table = load_project_partition(path)
    return table["project_code"].tolist(), table.loc[:, FEATURES].to_numpy(dtype=float)
