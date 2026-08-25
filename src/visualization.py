"""PCA e visualizações exploratórias, isolados da classificação.

As coordenadas desta unidade servem somente para apresentação. A decisão do
modelo continua sendo calculada em :mod:`src.classifier`, no espaço normalizado
completo das 42 características.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .preprocessing import normalize_rows, validate_vectors


@dataclass(frozen=True)
class PCAProjection:
    """Resultado auditável de uma projeção exclusivamente visual."""

    pca: PCA
    concepts: np.ndarray
    collections: tuple[np.ndarray, ...]

    @property
    def explained_variance_percent(self):
        return self.pca.explained_variance_ratio_ * 100

    def __iter__(self):
        """Preserva o desempacotamento da interface anterior."""
        yield self.pca
        yield self.concepts
        yield list(self.collections)


def project_pca(concepts, *collections, n_components=3, seed=42):
    """Ajusta PCA nos conceitos normalizados e projeta outras coleções.

    PCA não produz scores nem classes. Em particular, nenhuma decisão deve ser
    inferida das três coordenadas ou da região de dispersão.
    """
    if n_components != 3:
        raise ValueError("the 3D visualization requires exactly 3 components")
    concept_values = validate_vectors(concepts, name="concepts")
    normalized_concepts = normalize_rows(concept_values)
    if normalized_concepts.shape[0] < n_components:
        raise ValueError("PCA requires at least 3 concept points")
    pca = PCA(n_components=n_components, random_state=seed)
    concept_points = pca.fit_transform(normalized_concepts)
    projected = []
    for index, values in enumerate(collections):
        validated = validate_vectors(
            values,
            expected_columns=concept_values.shape[1],
            name=f"collection {index}",
        )
        projected.append(pca.transform(normalize_rows(validated)))
    projected = tuple(projected)
    return PCAProjection(pca, concept_points, projected)


def dispersion_ellipsoid(points, scale=2.0):
    """Retorna centro, raios e eixos de uma região descritiva de dispersão.

    ``scale`` multiplica os desvios-padrão principais. Isto não é um intervalo
    de confiança. Pequenos autovalores negativos introduzidos por arredondamento
    são truncados em zero antes da raiz quadrada.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("points must have shape (n, 3)")
    if values.shape[0] < 2:
        raise ValueError("at least 2 points are required for a dispersion region")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")

    center = values.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(values, rowvar=False))
    # eigh pode devolver -epsilon para matrizes semidefinidas positivas.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = eigenvalues.argsort()[::-1]
    radii = float(scale) * np.sqrt(eigenvalues[order])
    return center, radii, eigenvectors[:, order]


def plot_dispersion_ellipsoid_3d(
    ax, points, *, scale=2.0, color="yellow", alpha=0.18,
    edgecolor="goldenrod",
):
    """Desenha uma região de dispersão; não representa confiança estatística."""
    center, radii, eigenvectors = dispersion_ellipsoid(points, scale=scale)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    sphere = np.stack(
        [np.outer(np.cos(u), np.sin(v)),
         np.outer(np.sin(u), np.sin(v)),
         np.outer(np.ones_like(u), np.cos(v))],
        axis=-1,
    )
    surface = sphere * radii @ eigenvectors.T + center
    ax.plot_surface(*np.moveaxis(surface, 2, 0), color=color, alpha=alpha,
                    linewidth=0, shade=True)
    ax.plot_wireframe(*np.moveaxis(surface, 2, 0), color=edgecolor,
                      linewidth=0.3, alpha=0.20)
    return center, radii


def save_pca_plot(
    path, concepts, projects, class_names, *, data_version="não informada",
    model_version="não informada", project_label="Projetos",
):
    """Salva a projeção 3D com variância, limitação e versões visíveis."""
    import matplotlib.pyplot as plt

    projection = project_pca(concepts, projects)
    concept_points = projection.concepts
    (project_points,) = projection.collections
    variance = projection.explained_variance_percent
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*concept_points.T, marker="*", s=150, label="Classes")
    ax.scatter(*project_points.T, marker="x", label=project_label)
    for name, point in zip(class_names, concept_points):
        ax.text(*point, str(name), fontsize=8)
    if len(project_points) >= 2:
        plot_dispersion_ellipsoid_3d(ax, project_points)
    ax.set_xlabel(f"PC1 ({variance[0]:.2f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]:.2f}%)")
    ax.set_zlabel(f"PC3 ({variance[2]:.2f}%)")
    ax.set_title("PCA 3D normalizado — visualização exploratória")
    note = (
        f"Variância: PC1 {variance[0]:.2f}% | PC2 {variance[1]:.2f}% | "
        f"PC3 {variance[2]:.2f}% | total {variance.sum():.2f}%\n"
        "AVISO: a projeção em 3 componentes reduz informação; não é usada "
        "para classificar nem define confiança.\n"
        f"Dados: {data_version} | Modelo: {model_version}"
    )
    fig.text(0.02, 0.02, note, fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return concept_points, project_points
