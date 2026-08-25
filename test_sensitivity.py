import numpy as np
import pandas as pd

from src.sensitivity import (
    SensitivityConfiguration,
    controlled_configurations,
    run_sensitivity_analysis,
    select_configuration,
)


def _table(codes, labels, rows):
    data = {"project_code": codes, "real_class": labels}
    data.update({f"X{i:02d}": np.asarray(rows)[:, i - 1] for i in range(1, 43)})
    return pd.DataFrame(data)


def test_grade_e_relatorio_registram_metricas_e_instabilidade():
    concepts = np.zeros((2, 42))
    concepts[0, 0] = 1
    concepts[1, 1] = 1
    train = _table(["T1", "T2"], ["A", "A"], [[1] + [0] * 41, [1] + [0] * 41])
    validation = _table(
        ["V1", "V2"], ["A", "B"], [[1] + [0] * 41, [0, 1] + [0] * 40]
    )
    configs = [
        SensitivityConfiguration("reference"),
        SensitivityConfiguration("review", minimum_similarity=1.01),
    ]

    results, unstable = run_sensitivity_analysis(
        train, validation, concepts, ["A", "B"], configs
    )

    assert list(results["changed_decisions"]) == [0, 2]
    assert list(results["manual_review_count"]) == [0, 2]
    assert "security_f1" in results
    assert set(unstable["project_code"]) == {"V1", "V2"}
    assert select_configuration(results)["name"] == "reference"


def test_grade_controlada_muda_apenas_um_fator_fora_da_ordem():
    configurations = controlled_configurations(sequential_order_seeds=(3,))
    reference = configurations[0]
    for config in configurations[1:]:
        changed = [
            field
            for field in (
                "c32_weight", "c_extra_weight", "eta",
                "minimum_similarity", "minimum_margin", "sequential",
            )
            if getattr(config, field) != getattr(reference, field)
        ]
        assert len(changed) == 1
