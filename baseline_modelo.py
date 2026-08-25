"""Persistência compacta e auditável de baselines do modelo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _matrix_metadata(matrix: np.ndarray, version: str) -> dict:
    """Descreve uma matriz e identifica exatamente seu conteúdo."""
    contiguous = np.ascontiguousarray(matrix)
    return {
        "version": version,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _ranking_records(codes, classes, similarities):
    ranking = np.argsort(similarities, axis=1)[:, ::-1]
    records = []
    for row, code in enumerate(codes):
        first, second = ranking[row, :2]
        first_score = float(similarities[row, first])
        second_score = float(similarities[row, second])
        records.append({
            "project": code,
            "first_class": classes[first],
            "first_similarity": first_score,
            "second_class": classes[second],
            "second_similarity": second_score,
            "margin": first_score - second_score,
        })
    return records


def save_baseline(
    output_dir,
    *,
    concept_matrix_version,
    matrix_versions,
    classes,
    project_codes,
    validation_codes,
    c32,
    c_extra,
    projects,
    validation_projects,
    project_similarities,
    validation_similarities,
    conceptual_security_weights,
    normalized_conceptual_security_weights,
    hebbian_weights,
    pca_explained_variance,
):
    """Grava metadados legíveis e os vetores completos sem duplicá-los no código."""
    output_dir = Path(output_dir)
    baseline_dir = output_dir / "baselines" / concept_matrix_version
    baseline_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        "C32": np.asarray(c32),
        "C_EXTRA": np.asarray(c_extra),
        "P": np.asarray(projects),
        "V": np.asarray(validation_projects),
    }
    metadata = {
        "baseline_version": concept_matrix_version,
        "concept_matrix_version": concept_matrix_version,
        "matrices": {
            name: _matrix_metadata(value, matrix_versions[name])
            for name, value in matrices.items()
        },
        "project_vs_class": {
            "training_projects": _ranking_records(
                project_codes, classes, project_similarities
            ),
            "validation_projects": _ranking_records(
                validation_codes, classes, validation_similarities
            ),
            "all_similarities_storage": "baseline_arrays.npz",
        },
        "pca_explained_variance_percent": {
            "components": [float(value) for value in pca_explained_variance],
            "total": float(np.sum(pca_explained_variance)),
        },
        "weights_storage": {
            "conceptual_security": "baseline_arrays.npz:conceptual_security_weights",
            "normalized_conceptual_security": (
                "baseline_arrays.npz:normalized_conceptual_security_weights"
            ),
            "hebbian_prototype": "baseline_arrays.npz:hebbian_weights",
        },
        "quality_evidence": (
            "Not evaluated: the two validation projects both belong to Security; "
            "their classifications are diagnostics, not evidence of model quality."
        ),
    }

    metadata_path = baseline_dir / "baseline.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    arrays_path = baseline_dir / "baseline_arrays.npz"
    np.savez_compressed(
        arrays_path,
        **matrices,
        project_similarities=project_similarities,
        validation_similarities=validation_similarities,
        conceptual_security_weights=conceptual_security_weights,
        normalized_conceptual_security_weights=normalized_conceptual_security_weights,
        hebbian_weights=hebbian_weights,
        pca_explained_variance_percent=pca_explained_variance,
    )
    return metadata_path, arrays_path
