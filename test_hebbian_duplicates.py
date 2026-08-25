import numpy as np

from src.hebbian import (
    apply_hebbian,
    historical_prototype,
    sequential_seed_variation,
    train_class_prototypes,
    train_hebbian,
)


def test_duplicata_nao_altera_h_nem_w():
    projects = np.array([[1.0, 0.0], [0.0, 1.0]])
    duplicated = np.vstack([projects, projects[0]])
    np.testing.assert_allclose(historical_prototype(projects), historical_prototype(duplicated))
    original, _ = train_hebbian(projects, [1.0, 1.0])
    repeated, history = train_hebbian(duplicated, [1.0, 1.0])
    np.testing.assert_allclose(original, repeated)
    assert len(history) == 3


def test_atualizacao_agregada_independe_da_ordem():
    projects = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    forward, _ = train_hebbian(projects, [1.0, 0.0], epochs=3)
    reverse, _ = train_hebbian(projects[::-1], [1.0, 0.0], epochs=3)
    np.testing.assert_allclose(forward, reverse)


def test_treino_multiclasse_cria_um_prototipo_por_classe():
    projects = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = ["A", "A", "B", "B"]
    prototypes = train_class_prototypes(
        projects, labels, ["A", "B"], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert prototypes.shape == (2, 2)
    assert apply_hebbian(projects, prototypes).shape == (4, 2)


def test_treino_sequencial_reporta_variacao_entre_sementes():
    variation = sequential_seed_variation(
        [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], [7, 11]
    )
    assert variation.seeds == (7, 11)
    assert variation.prototypes.shape == (2, 2)
    assert 0 <= variation.pairwise_cosine_min <= variation.pairwise_cosine_max <= 1
