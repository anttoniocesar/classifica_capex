import json

import numpy as np

from baseline_modelo import save_baseline


def test_save_baseline_registra_ranking_pesos_e_matrizes(tmp_path):
    similarities = np.array([[0.2, 0.8, 0.5]])
    metadata_path, arrays_path = save_baseline(
        tmp_path,
        concept_matrix_version="1.0",
        matrix_versions={name: "1.0" for name in ("C32", "C_EXTRA", "P", "V")},
        classes=["A", "B", "C"],
        project_codes=["P1"],
        validation_codes=["V1"],
        c32=np.ones((3, 2)),
        c_extra=np.zeros((3, 1)),
        projects=np.ones((1, 3)),
        validation_projects=np.ones((1, 3)),
        project_similarities=similarities,
        validation_similarities=similarities,
        conceptual_security_weights=np.array([0.1, 0.2, 0.3]),
        normalized_conceptual_security_weights=np.array([0.1, 0.2, 0.3]),
        hebbian_weights=np.array([0.2, 0.3, 0.4]),
        pca_explained_variance=np.array([50.0, 25.0, 10.0]),
    )

    baseline = json.loads(metadata_path.read_text(encoding="utf-8"))
    ranking = baseline["project_vs_class"]["training_projects"][0]
    assert (ranking["first_class"], ranking["second_class"]) == ("B", "C")
    assert np.isclose(ranking["margin"], 0.3)
    assert baseline["pca_explained_variance_percent"]["total"] == 85.0
    assert baseline["matrices"]["C32"]["shape"] == [3, 2]

    with np.load(arrays_path) as arrays:
        assert set(("C32", "C_EXTRA", "P", "V")) <= set(arrays.files)
        np.testing.assert_array_equal(arrays["project_similarities"], similarities)
