"""Esquema canônico das características, classes e matrizes."""
import numpy as np
from modelo_seguranca_pca_hebb import FEATURES, CLASSES


def validate_structure(concepts, projects=None):
    concepts = np.asarray(concepts, dtype=float)
    expected = (len(CLASSES), len(FEATURES))
    if concepts.shape != expected:
        raise ValueError(f"matriz conceitual deve ter formato {expected}, recebeu {concepts.shape}")
    if projects is not None:
        projects = np.asarray(projects, dtype=float)
        if projects.ndim != 2 or projects.shape[1] != len(FEATURES):
            raise ValueError(f"projetos devem possuir {len(FEATURES)} características")
    return True
