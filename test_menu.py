import numpy as np
import matplotlib.pyplot as plt

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


def test_menu_cria_figura_com_eixo_3d_sem_janela_tk(tmp_path):
    figure, path = MenuService(tmp_path).create_3d_chart()

    try:
        assert path is None
        assert len(figure.axes) == 1
        assert figure.axes[0].name == "3d"
    finally:
        plt.close(figure)


def test_menu_exporta_grafico_3d_para_png_sem_janela_tk(tmp_path):
    figure, path = MenuService(tmp_path).create_3d_chart(export=True)

    try:
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        plt.close(figure)
