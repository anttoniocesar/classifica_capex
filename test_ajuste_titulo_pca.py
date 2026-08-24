from ajuste_titulo_pca import definir_titulo_pca


class EixoFalso:
    def __init__(self):
        self.chamada = None

    def set_title(self, titulo, **opcoes):
        self.chamada = (titulo, opcoes)


def test_definir_titulo_pca_configura_texto_e_formatacao():
    eixo = EixoFalso()

    definir_titulo_pca(eixo)

    assert eixo.chamada == (
        "PCA 3D Normalizado — Espaço Associativo das 13 Classes "
        "+ 19 Projetos de Segurança",
        {"fontsize": 18, "fontweight": "bold", "pad": 24},
    )
