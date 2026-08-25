"""Carregamento explícito das matrizes conceituais e dos projetos."""
from pathlib import Path
import pandas as pd
import numpy as np
from modelo_seguranca_pca_hebb import C32, C_EXTRA
from .schema import FEATURES, CLASSES, validate_structure


def load_concept_matrices():
    """Retorna cópias das matrizes base e da matriz conceitual completa."""
    c32, extra = C32.copy(), C_EXTRA.copy()
    concepts = np.hstack((c32, extra))
    validate_structure(concepts)
    return c32, extra, concepts


def load_projects(workbook=Path("resultados/resultados_modelo_seguranca.xlsx"), sheet="Projetos_Seguranca"):
    """Carrega projetos de uma planilha gerada pelo fluxo, somente sob chamada."""
    table = pd.read_excel(workbook, sheet_name=sheet, index_col=0)
    missing = [feature for feature in FEATURES if feature not in table.columns]
    if missing:
        raise ValueError(f"características ausentes: {missing}")
    projects = table.loc[:, FEATURES].to_numpy(dtype=float)
    validate_structure(np.hstack((C32, C_EXTRA)), projects)
    return table.index.astype(str).tolist(), projects
