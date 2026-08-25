"""PCA, elipsoide de confiança e gráficos; nada é exibido ao importar."""
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from .preprocessing import normalize_rows


def project_pca(concepts, *collections, n_components=3, seed=42):
    pca = PCA(n_components=n_components, random_state=seed)
    concept_points = pca.fit_transform(normalize_rows(concepts))
    projected = [pca.transform(normalize_rows(values)) for values in collections]
    return pca, concept_points, projected


def confidence_ellipsoid(points, confidence=0.95):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False) + np.eye(3) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    radii = np.sqrt(chi2.ppf(confidence, df=3)) * np.sqrt(eigenvalues[order])
    return center, radii, eigenvectors[:, order]


def save_pca_plot(path, concepts, projects, class_names):
    import matplotlib.pyplot as plt
    _, concept_points, (project_points,) = project_pca(concepts, projects)
    fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*concept_points.T, label="Classes")
    ax.scatter(*project_points.T, marker="x", label="Projetos")
    for name, point in zip(class_names, concept_points): ax.text(*point, name)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return concept_points, project_points
