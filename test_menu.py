import numpy as np

from src.menu import MenuService
from src.schema import FEATURES


def test_menu_classifica_um_projeto():
    values = {feature: 0.0 for feature in FEATURES}
    values[FEATURES[0]] = 1.0
    result = MenuService().classify("NOVO-001", values)

    assert result.code == "NOVO-001"
    assert result.predicted_class
    assert 0 <= result.similarity <= 1


def test_menu_treina_e_persiste_prototipo(tmp_path):
    result = MenuService(tmp_path).train()

    assert result.path.is_file()
    artifact = np.load(result.path)
    assert artifact["weights"].shape == (len(FEATURES),)
    assert artifact["project_count"] == result.project_count


def test_menu_gera_grafico_3d(tmp_path):
    path = MenuService(tmp_path).create_3d_chart()

    assert path.is_file()
    assert path.stat().st_size > 0
