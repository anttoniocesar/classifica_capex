"""Esquema canônico das características, classes e matrizes."""
import numpy as np

from modelo_seguranca_pca_hebb import CLASSES, FEATURES


FEATURE_MIN = 0.0
FEATURE_MAX = 1.0


def validate_feature_values(values, context="dados"):
    """Valida dimensão, finitude, escala e vetores não nulos."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError(f"{context}: deve possuir exatamente {len(FEATURES)} características")
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: características devem conter somente valores finitos")
    if ((values < FEATURE_MIN) | (values > FEATURE_MAX)).any():
        raise ValueError(
            f"{context}: características devem estar na escala [{FEATURE_MIN}, {FEATURE_MAX}]"
        )
    zero_rows = np.flatnonzero(np.all(values == 0, axis=1)).tolist()
    if zero_rows:
        raise ValueError(f"{context}: vetores totalmente zerados nas linhas {zero_rows}")
    return True


def validate_structure(concepts, projects=None):
    concepts = np.asarray(concepts, dtype=float)
    expected = (len(CLASSES), len(FEATURES))
    if concepts.shape != expected:
        raise ValueError(f"matriz conceitual deve ter formato {expected}, recebeu {concepts.shape}")
    validate_feature_values(concepts, "matriz conceitual")
    if projects is not None:
        validate_feature_values(projects, "projetos")
    return True
