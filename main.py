"""Ponto de entrada do baseline conceitual oficial de projetos CAPEX."""

from src.classifier import classify
from src.config import MIN_CONFIDENCE, MIN_MARGIN
from src.data import load_concept_matrices, load_project_partitions
from src.schema import CLASSES, FEATURES


def run_conceptual_baseline():
    """Classifica as partições diretamente contra a matriz conceitual X01–X42."""
    _, _, class_concept_matrix = load_concept_matrices()
    partitions = load_project_partitions()
    results = {}
    for partition_name, table in partitions.items():
        project_feature_matrix = table.loc[:, FEATURES].to_numpy(dtype=float)
        results[partition_name] = classify(
            project_feature_matrix,
            class_concept_matrix,
            CLASSES,
            project_codes=table["project_code"],
            min_confidence=MIN_CONFIDENCE,
            min_margin=MIN_MARGIN,
        )
    return results


def main():
    """Executa o baseline oficial; efeitos de saída ocorrem apenas nesta chamada."""
    results = run_conceptual_baseline()
    for partition_name, (_, decisions) in results.items():
        for decision in decisions:
            print(partition_name, decision)


if __name__ == "__main__":
    main()
