"""Auditoria de prontidão do corpus real, separada da matriz conceitual."""
from collections import Counter

from .data import find_cross_partition_feature_duplicates
from .schema import CLASSES


REQUIRED_BOUNDARIES = {
    "seguranca_manutencao",
    "seguranca_renovacoes",
    "seguranca_meio_ambiente",
    "seguranca_rejeitos",
    "seguranca_modificacoes",
    "legal_nao_seguranca",
}


def audit_dataset(partitions, registry):
    """Produz pendências sem promover dados incompletos a conjunto final.

    ``registry`` deve conter uma entrada por código com ``project_family``,
    ``used_for_concept_matrix`` e ``boundary_cases``. Famílias representam o
    mesmo investimento e, portanto, jamais podem atravessar partições.
    """
    issues = []
    rows = [(split, row) for split, table in partitions.items() for _, row in table.iterrows()]
    codes = {row["project_code"] for _, row in rows}
    missing = sorted(codes - set(registry))
    if missing:
        issues.append({"type": "missing_registry", "projects": missing})

    eligible = []
    families = {}
    boundaries = Counter()
    for split, row in rows:
        code = row["project_code"]
        metadata = registry.get(code)
        if not metadata:
            continue
        if metadata.get("used_for_concept_matrix") is not False:
            issues.append({"type": "concept_matrix_reuse_or_unknown", "project": code})
            continue
        family = str(metadata.get("project_family", "")).strip()
        if not family:
            issues.append({"type": "missing_project_family", "project": code})
            continue
        families.setdefault(family, set()).add(split)
        eligible.append(row["real_class"])
        for boundary in metadata.get("boundary_cases", []):
            boundaries[boundary] += 1

    for family, splits in sorted(families.items()):
        if len(splits) > 1:
            issues.append({"type": "family_leakage", "family": family, "splits": sorted(splits)})

    missing_classes = sorted(set(CLASSES) - set(eligible))
    if missing_classes:
        issues.append({"type": "missing_classes", "classes": missing_classes})
    missing_boundaries = sorted(REQUIRED_BOUNDARIES - set(boundaries))
    if missing_boundaries:
        issues.append({"type": "missing_boundary_cases", "boundaries": missing_boundaries})

    duplicates = find_cross_partition_feature_duplicates(partitions)
    if duplicates:
        issues.append({"type": "identical_vectors_across_splits", "groups": duplicates})
    return {
        "ready": not issues,
        "eligible_projects": len(eligible),
        "class_counts": dict(sorted(Counter(eligible).items())),
        "boundary_counts": dict(sorted(boundaries.items())),
        "issues": issues,
    }
