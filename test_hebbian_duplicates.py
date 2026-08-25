import numpy as np

from src.hebbian import historical_prototype, train_hebbian


def test_duplicata_nao_altera_h_nem_w():
    projects = np.array([[1.0, 0.0], [0.0, 1.0]])
    duplicated = np.vstack([projects, projects[0]])
    np.testing.assert_allclose(historical_prototype(projects), historical_prototype(duplicated))
    original, _ = train_hebbian(projects, [1.0, 1.0])
    repeated, history = train_hebbian(duplicated, [1.0, 1.0])
    np.testing.assert_allclose(original, repeated)
    assert len(history) == 3
