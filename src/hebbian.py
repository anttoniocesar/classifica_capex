"""Protótipos Hebbianos direcionais, reproduzíveis e multiclasse.

Definição matemática
--------------------
Para vetores de projeto ``x_i != 0``, escrevemos ``n(x)=x/||x||_2``. O vetor
inicial é sempre o conceito da própria classe, ``w^(0)=n(c)``. No modo
agregado (padrão), uma época aplica

``w^(t+1) = n(w^(t) + eta * sum_i n(x_i))``.

Assim, a atualização não depende da ordem das linhas. ``eta`` é a taxa de
aprendizagem (``0.05`` por padrão), a normalização L2 ocorre no vetor inicial,
em cada exemplo e depois de cada época, e ``epochs`` vale 1 por padrão. O treino
para antecipadamente quando ``||w^(t+1)-w^(t)||_2 <= tolerance``; ``tolerance=0``
desativa essa parada. Duplicatas exatas têm peso único por padrão.

O modo sequencial, mantido apenas para experimentos, aplica
``w <- n(w + eta*n(x_i))`` depois de embaralhar cada época com um gerador de
semente explícita. :func:`sequential_seed_variation` reporta a variação obtida.

Na classificação, aprende-se **um protótipo por classe** e escolhe-se
``argmax_k n(x) . w_k``. Um único vetor positivo não é um classificador das 13
categorias; :func:`train_class_prototypes` exige amostras de todas as classes.
"""

from dataclasses import dataclass

import numpy as np

from .preprocessing import normalize_vector


@dataclass(frozen=True)
class SeedVariation:
    """Resumo da sensibilidade do treino sequencial às sementes."""

    seeds: tuple
    prototypes: np.ndarray
    pairwise_cosine_min: float
    pairwise_cosine_max: float
    maximum_l2_distance: float


def unique_vectors(projects):
    """Mantém uma ocorrência de cada representação, em ordem estável."""
    values = np.asarray(projects, dtype=float)
    if values.ndim != 2:
        raise ValueError("projects deve ser uma matriz bidimensional")
    if not len(values):
        return values.copy(), np.empty(0, dtype=int)
    _, first, inverse = np.unique(values, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    return values[first[order]].copy(), np.bincount(remap[inverse])


def _normalized_rows(projects):
    values = np.asarray(projects, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("projects deve ser uma matriz bidimensional não vazia")
    return np.asarray([normalize_vector(row) for row in values])


def historical_prototype(projects):
    """Calcula ``H`` dando o mesmo peso a cada vetor distinto."""
    distinct, _ = unique_vectors(projects)
    if not len(distinct):
        raise ValueError("projects não pode ser vazio")
    return distinct.mean(axis=0)


def train_hebbian(
    projects,
    initial_prototype,
    eta=0.05,
    *,
    epochs=1,
    tolerance=0.0,
    mode="aggregate",
    seed=0,
    deduplicate=True,
):
    """Treina um protótipo segundo a regra matemática descrita no módulo.

    ``history`` contém o vetor inicial e os estados de fim de época. Para
    compatibilidade, no caso agregado de uma época ele repete o estado final
    uma vez por amostra; isso não representa atualizações sequenciais.
    """
    if eta <= 0 or not np.isfinite(eta):
        raise ValueError("eta deve ser positivo e finito")
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs deve ser um inteiro positivo")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance deve ser não negativo e finito")
    if mode not in {"aggregate", "sequential"}:
        raise ValueError("mode deve ser 'aggregate' ou 'sequential'")

    training = (
        unique_vectors(projects)[0]
        if deduplicate
        else np.asarray(projects, dtype=float)
    )
    normalized = _normalized_rows(training)
    weights = normalize_vector(initial_prototype)
    history = [weights.copy()]
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        previous = weights.copy()
        if mode == "aggregate":
            weights = normalize_vector(weights + eta * normalized.sum(axis=0))
        else:
            for index in rng.permutation(len(normalized)):
                weights = normalize_vector(weights + eta * normalized[index])
        history.append(weights.copy())
        if tolerance and np.linalg.norm(weights - previous) <= tolerance:
            break

    # A API antiga expunha um estado por vetor com epochs=1.
    if mode == "aggregate" and epochs == 1 and len(normalized) > 1:
        history.extend([weights.copy()] * (len(normalized) - 1))
    return weights, np.asarray(history)


def train_class_prototypes(
    projects, labels, class_names, initial_prototypes, **training_options
):
    """Aprende uma linha por classe; nunca reutiliza um positivo para todas."""
    values = np.asarray(projects, dtype=float)
    labels = np.asarray(labels, dtype=object)
    classes = list(class_names)
    initial = np.asarray(initial_prototypes, dtype=float)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("projects e labels têm formatos incompatíveis")
    if initial.shape != (len(classes), values.shape[1]):
        raise ValueError("initial_prototypes deve ter uma linha por classe")
    unknown = sorted(set(labels) - set(classes))
    missing = [name for name in classes if name not in set(labels)]
    if unknown or missing:
        raise ValueError(f"rótulos desconhecidos={unknown}; classes sem amostras={missing}")
    return np.asarray(
        [
            train_hebbian(
                values[labels == name], initial[index], **training_options
            )[0]
            for index, name in enumerate(classes)
        ]
    )


def sequential_seed_variation(projects, initial_prototype, seeds, **training_options):
    """Treina sequencialmente e resume a variação entre sementes informadas."""
    seeds = tuple(seeds)
    if len(seeds) < 2:
        raise ValueError("informe ao menos duas sementes")
    options = dict(training_options)
    options["mode"] = "sequential"
    prototypes = np.asarray(
        [
            train_hebbian(projects, initial_prototype, seed=seed, **options)[0]
            for seed in seeds
        ]
    )
    cosines = prototypes @ prototypes.T
    distances = np.linalg.norm(prototypes[:, None] - prototypes[None, :], axis=2)
    triangle = np.triu_indices(len(seeds), 1)
    return SeedVariation(
        seeds,
        prototypes,
        float(cosines[triangle].min()),
        float(cosines[triangle].max()),
        float(distances.max()),
    )


def apply_hebbian(vectors, weights):
    """Retorna cossenos; com matriz de pesos, retorna projeto x classe."""
    projects = _normalized_rows(vectors)
    prototypes = np.asarray(weights, dtype=float)
    if prototypes.ndim == 1:
        return projects @ normalize_vector(prototypes)
    if prototypes.ndim != 2:
        raise ValueError("weights deve ser vetor ou matriz")
    normalized = np.asarray([normalize_vector(row) for row in prototypes])
    if projects.shape[1] != normalized.shape[1]:
        raise ValueError("vectors e weights devem ter o mesmo número de colunas")
    return projects @ normalized.T


def classify_hebbian(vectors, class_prototypes, class_names):
    """Classifica por ``argmax`` entre os protótipos aprendidos por classe."""
    scores = apply_hebbian(vectors, class_prototypes)
    classes = list(class_names)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("class_names deve identificar cada protótipo")
    return np.asarray([classes[index] for index in scores.argmax(axis=1)]), scores
