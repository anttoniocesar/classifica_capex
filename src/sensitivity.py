"""Análise controlada de sensibilidade sem acesso ao teste final.

O experimento altera um fator por vez em relação à configuração de referência.
Os pesos de ``C32`` e ``C_EXTRA`` são pesos de atributos: eles são aplicados
tanto aos projetos quanto aos conceitos antes do cosseno.  ``eta`` atualiza
somente o protótipo de Segurança; as demais classes continuam conceituais.
"""

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from .classifier import produce_decisions
from .evaluation import evaluate
from .hebbian import apply_hebbian, train_hebbian


@dataclass(frozen=True)
class SensitivityConfiguration:
    """Parâmetros reproduzíveis de uma execução da análise."""

    name: str
    c32_weight: float = 1.0
    c_extra_weight: float = 1.0
    eta: float = 0.05
    minimum_similarity: float = 0.0
    minimum_margin: float = 0.0
    sequential: bool = False
    order_seed: int = 0


def controlled_configurations(
    *,
    c32_weights=(0.75, 1.0, 1.25),
    c_extra_weights=(0.75, 1.0, 1.25),
    etas=(0.025, 0.05, 0.1),
    minimum_similarities=(0.0, 0.5, 0.7),
    minimum_margins=(0.0, 0.05, 0.1),
    sequential_order_seeds=(0, 1, 2),
):
    """Cria uma grade OFAT (um fator por vez), incluindo a referência."""
    reference = SensitivityConfiguration("reference")
    configs = [reference]
    factors = (
        ("c32_weight", c32_weights),
        ("c_extra_weight", c_extra_weights),
        ("eta", etas),
        ("minimum_similarity", minimum_similarities),
        ("minimum_margin", minimum_margins),
    )
    for field, values in factors:
        for value in values:
            if value != getattr(reference, field):
                configs.append(
                    SensitivityConfiguration(
                        name=f"{field}={value}", **{field: float(value)}
                    )
                )
    for seed in sequential_order_seeds:
        configs.append(
            SensitivityConfiguration(
                name=f"sequential_order_seed={seed}",
                sequential=True,
                order_seed=int(seed),
            )
        )
    return configs


def _validate_configuration(config):
    for name in ("c32_weight", "c_extra_weight", "eta"):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} deve ser positivo e finito")
    for name in ("minimum_similarity", "minimum_margin"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} deve ser não negativo e finito")


def _scores(training_projects, evaluation_projects, concepts, config):
    _validate_configuration(config)
    feature_weights = np.r_[
        np.full(32, config.c32_weight), np.full(10, config.c_extra_weight)
    ]
    weighted_train = np.asarray(training_projects, dtype=float) * feature_weights
    weighted_evaluation = np.asarray(evaluation_projects, dtype=float) * feature_weights
    weighted_concepts = np.asarray(concepts, dtype=float) * feature_weights
    security, _ = train_hebbian(
        weighted_train,
        weighted_concepts[0],
        eta=config.eta,
        mode="sequential" if config.sequential else "aggregate",
        seed=config.order_seed,
    )
    prototypes = weighted_concepts.copy()
    prototypes[0] = security
    return apply_hebbian(weighted_evaluation, prototypes)


def run_sensitivity_analysis(
    training_table,
    evaluation_table,
    concepts,
    class_names,
    configurations,
):
    """Avalia configurações em desenvolvimento/validação e resume instabilidade.

    A função deliberadamente recebe tabelas, não nomes de partições, e nunca
    carrega ``projects_test.csv``. O chamador deve fornecer desenvolvimento no
    primeiro argumento e validação no segundo.
    """
    feature_columns = [column for column in training_table if column.startswith("X")]
    if len(feature_columns) != 42:
        raise ValueError("training_table deve conter as 42 características X01–X42")
    if list(column for column in evaluation_table if column.startswith("X")) != feature_columns:
        raise ValueError("desenvolvimento e validação devem usar o mesmo esquema")
    configurations = list(configurations)
    if not configurations:
        raise ValueError("informe ao menos uma configuração")

    security_training = training_table[
        training_table["real_class"] == list(class_names)[0]
    ]
    if security_training.empty:
        raise ValueError("desenvolvimento não contém projetos de Segurança")
    train_x = security_training.loc[:, feature_columns].to_numpy(dtype=float)
    eval_x = evaluation_table.loc[:, feature_columns].to_numpy(dtype=float)
    codes = evaluation_table["project_code"].astype(str).tolist()
    truth = evaluation_table["real_class"].tolist()
    runs = []
    decision_keys = []
    for config in configurations:
        scores = _scores(train_x, eval_x, concepts, config)
        decisions = produce_decisions(
            codes,
            class_names,
            scores,
            minimum_similarity=config.minimum_similarity,
            minimum_margin=config.minimum_margin,
        )
        review = np.asarray([decision.review_required for decision in decisions])
        predictions = np.asarray([decision.closest_class for decision in decisions])
        metrics = evaluate(truth, predictions, class_names, review_mask=review)
        keys = [(decision.closest_class, decision.status) for decision in decisions]
        decision_keys.append(keys)
        runs.append((config, decisions, metrics))

    reference_keys = decision_keys[0]
    records = []
    for (config, decisions, metrics), keys in zip(runs, decision_keys):
        all_metrics = metrics["all_samples"]
        security = all_metrics["security"]
        record = {
            **asdict(config),
            **{f"security_{key}": value for key, value in security.items()},
            "macro_precision": all_metrics["macro_precision"],
            "macro_recall": all_metrics["macro_recall"],
            "macro_f1": all_metrics["macro_f1"],
            "balanced_accuracy": all_metrics["balanced_accuracy"],
            "automatic_macro_precision": metrics["automatic_decisions"]["macro_precision"],
            "automatic_macro_recall": metrics["automatic_decisions"]["macro_recall"],
            "automatic_macro_f1": metrics["automatic_decisions"]["macro_f1"],
            "changed_decisions": sum(a != b for a, b in zip(reference_keys, keys)),
            "manual_review_count": metrics["manual_review_count"],
            "automatic_coverage": metrics["automatic_coverage"],
        }
        records.append(record)

    unstable = []
    for index, code in enumerate(codes):
        outcomes = {keys[index] for keys in decision_keys}
        if len(outcomes) > 1:
            unstable.append(
                {
                    "project_code": code,
                    "outcome_count": len(outcomes),
                    "outcomes": " | ".join(
                        f"{class_name} [{status}]"
                        for class_name, status in sorted(outcomes)
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(
        unstable, columns=["project_code", "outcome_count", "outcomes"]
    )


def select_configuration(results):
    """Seleciona somente a partir do resultado de desenvolvimento/validação.

    Prioriza F1 de Segurança, F1 macro e cobertura, nesta ordem; complexidade
    não é premiada implicitamente e o primeiro empate estável é preservado.
    """
    if results.empty:
        raise ValueError("results não pode ser vazio")
    ranking = results.sort_values(
        ["security_f1", "macro_f1", "automatic_coverage"],
        ascending=False,
        kind="stable",
    )
    return ranking.iloc[0]
