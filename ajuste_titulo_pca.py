"""Ajuste do título do gráfico PCA 3D do modelo de segurança."""


def definir_titulo_pca(ax):
    """Define no eixo 3D o título normalizado do espaço associativo."""
    ax.set_title(
        "PCA 3D Normalizado — Espaço Associativo das 13 Classes "
        "+ 19 Projetos de Segurança",
        fontsize=18,
        fontweight="bold",
        pad=24,
    )
