"""Auditoria de prontidão e de redundância do corpus de projetos."""
from collections import Counter

import numpy as np

from .data import find_cross_partition_feature_duplicates
from .schema import CLASSES, FEATURES


REQUIRED_BOUNDARIES = {
    "seguranca_manutencao",
    "seguranca_renovacoes",
    "seguranca_meio_ambiente",
    "seguranca_rejeitos",
    "seguranca_modificacoes",
    "legal_nao_seguranca",
}

DUPLICATE_DECISIONS = {
    "same_project",
    "different_projects_incomplete_data",
    "equivalent_projects",
    "insufficient_coding_granularity",
}


def _project_rows(partitions):
    return [
        (split, str(row["project_code"]), row)
        for split, table in partitions.items()
        for _, row in table.iterrows()
    ]


def audit_feature_vectors(partitions, registry=None, *, near_similarity=0.99):
    """Resume vetores idênticos/quase idênticos e sua revisão de negócio.

    ``duplicate_decision`` e ``duplicate_justification`` no registro de cada
    projeto documentam a conclusão humana. Similaridade de cosseno apenas gera
    candidatos: não decide que dois investimentos são a mesma coisa.
    """
    if not 0 < near_similarity <= 1:
        raise ValueError("near_similarity deve estar no intervalo (0, 1]")
    registry = registry or {}
    rows = _project_rows(partitions)
    values = np.asarray([[float(row[name]) for name in FEATURES] for _, _, row in rows])
    codes = [code for _, code, _ in rows]

    groups = {}
    for index, row in enumerate(values):
        groups.setdefault(row.astype("<f8").tobytes(), []).append(index)
    ordered_groups = sorted(groups.values(), key=lambda group: group[0])
    vector_frequency = [
        {
            "vector_id": f"V{number:04d}",
            "frequency": len(group),
            "projects": [codes[index] for index in group],
        }
        for number, group in enumerate(ordered_groups, 1)
    ]

    exact_duplicates = []
    for item, group in zip(vector_frequency, ordered_groups):
        if len(group) < 2:
            continue
        project_decisions = [registry.get(codes[index], {}).get("duplicate_decision") for index in group]
        project_justifications = [
            str(registry.get(codes[index], {}).get("duplicate_justification", "")).strip()
            for index in group
        ]
        decisions = {decision for decision in project_decisions if decision is not None}
        justifications = {value for value in project_justifications if value}
        resolved = (
            len(decisions) == 1
            and decisions <= DUPLICATE_DECISIONS
            and all(project_decisions)
            and all(project_justifications)
        )
        exact_duplicates.append({
            **item,
            "decision": next(iter(decisions)) if len(decisions) == 1 else None,
            "justification": next(iter(justifications)) if len(justifications) == 1 else None,
            "review_status": "resolved" if resolved else "pending_human_review",
        })

    near_duplicates = []
    norms = np.linalg.norm(values, axis=1) if len(values) else np.empty(0)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left].tobytes() == values[right].tobytes():
                continue
            similarity = float(np.dot(values[left], values[right]) / (norms[left] * norms[right]))
            if similarity >= near_similarity:
                different = [FEATURES[i] for i in np.flatnonzero(values[left] != values[right])]
                near_duplicates.append({
                    "projects": [codes[left], codes[right]],
                    "cosine_similarity": similarity,
                    "different_features": different,
                    "review_status": "candidate_for_human_review",
                })

    nonzero = values != 0
    distinct = [len(np.unique(values[:, i])) for i in range(len(FEATURES))] if len(values) else [0] * len(FEATURES)
    return {
        "total_projects": len(rows),
        "unique_vectors": len(ordered_groups),
        "vector_frequency": vector_frequency,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "constant_features": [FEATURES[i] for i, count in enumerate(distinct) if count == 1],
        "never_filled_features": [FEATURES[i] for i in range(len(FEATURES)) if len(values) and not nonzero[:, i].any()],
        "almost_always_present_features": [
            {"feature": FEATURES[i], "frequency": int(nonzero[:, i].sum()), "proportion": float(nonzero[:, i].mean())}
            for i in range(len(FEATURES))
            if len(values) and nonzero[:, i].mean() >= 0.9
        ],
        "near_similarity_threshold": near_similarity,
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

    vector_audit = audit_feature_vectors(partitions, registry)
    duplicates = find_cross_partition_feature_duplicates(partitions)
    if duplicates:
        issues.append({"type": "identical_vectors_across_splits", "groups": duplicates})
    if any(group["review_status"] != "resolved" for group in vector_audit["exact_duplicates"]):
        issues.append({"type": "identical_vectors_pending_review"})
    return {
        "ready": not issues,
        "eligible_projects": len(eligible),
        "class_counts": dict(sorted(Counter(eligible).items())),
        "boundary_counts": dict(sorted(boundaries.items())),
        "issues": issues,
        "vector_audit": vector_audit,
    }
