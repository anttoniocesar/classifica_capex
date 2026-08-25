import pandas as pd

from src.data import PROJECT_PARTITIONS, load_project_partitions
from src.dataset_audit import REQUIRED_BOUNDARIES, audit_dataset
from src.schema import CLASSES


def test_corpus_legado_nao_e_promovido_a_corpus_independente():
    report = audit_dataset(load_project_partitions(), registry={})
    assert report["ready"] is False
    assert report["eligible_projects"] == 0
    assert {issue["type"] for issue in report["issues"]} >= {
        "missing_registry", "missing_classes", "missing_boundary_cases"
    }


def test_familia_nao_pode_vazar_entre_particoes():
    source = pd.read_csv(PROJECT_PARTITIONS["train"]).iloc[[0]]
    copy = source.copy()
    copy.loc[:, "project_code"] = "OUTRO-CODIGO"
    partitions = {"train": source, "validation": copy, "test": source.iloc[0:0]}
    registry = {
        source.iloc[0]["project_code"]: {
            "project_family": "familia-1", "used_for_concept_matrix": False,
            "boundary_cases": sorted(REQUIRED_BOUNDARIES),
        },
        "OUTRO-CODIGO": {
            "project_family": "familia-1", "used_for_concept_matrix": False,
            "boundary_cases": [],
        },
    }
    report = audit_dataset(partitions, registry)
    issue_types = {issue["type"] for issue in report["issues"]}
    assert "family_leakage" in issue_types
    assert "identical_vectors_across_splits" in issue_types
    assert set(report["class_counts"]) <= set(CLASSES)
