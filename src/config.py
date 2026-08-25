"""Configuração central do fluxo, sem efeitos colaterais na importação."""
from pathlib import Path

OUTPUT_DIR = Path("resultados")
RANDOM_SEED = 42
ETA = 0.05
MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.05
CONCEPT_MATRIX_VERSION = "1.0"
MATRIX_VERSIONS = {name: "1.0" for name in ("C32", "C_EXTRA", "P", "V")}
