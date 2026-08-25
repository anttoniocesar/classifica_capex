from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    PROJECT_PARTITIONS,
    load_concept_matrices,
    load_project_partition,
    load_project_partitions,
)
from src.schema import CLASSES, FEATURES


def test_dados_versionados_sao_validos():
    _, _, concepts = load_concept_matrices()
    partitions = load_project_partitions()

    assert concepts.shape == (len(CLASSES), len(FEATURES))
    assert set(partitions) == {"train", "validation", "test"}
    assert len(partitions["train"]) == 19
    assert len(partitions["validation"]) == 2


def test_rejeita_vetor_totalmente_zerado(tmp_path):
    table = pd.read_csv(PROJECT_PARTITIONS["train"])
    table.loc[0, FEATURES] = 0
    path = tmp_path / "invalid.csv"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="totalmente zerados"):
        load_project_partition(path)


def test_rejeita_projeto_repetido_entre_particoes(tmp_path):
    paths = {}
    for name, source in PROJECT_PARTITIONS.items():
        target = tmp_path / source.name
        target.write_bytes(Path(source).read_bytes())
        paths[name] = target

    train = pd.read_csv(paths["train"])
    validation = pd.read_csv(paths["validation"])
    validation.loc[0, "project_code"] = train.loc[0, "project_code"]
    validation.to_csv(paths["validation"], index=False)

    with pytest.raises(ValueError, match="mais de uma partição"):
        load_project_partitions(paths)
