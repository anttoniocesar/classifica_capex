"""Executa e registra a análise controlada usando apenas treino e validação."""

from pathlib import Path

from src.data import PROJECT_PARTITIONS, load_concept_matrices, load_project_partition
from src.schema import CLASSES
from src.sensitivity import (
    controlled_configurations,
    run_sensitivity_analysis,
    select_configuration,
)


def main(output_dir=Path("resultados/sensibilidade")):
    # Não carregamos o teste: nem mesmo sua presença é necessária para escolher.
    training = load_project_partition(PROJECT_PARTITIONS["train"])
    validation = load_project_partition(PROJECT_PARTITIONS["validation"])
    _, _, concepts = load_concept_matrices()
    results, unstable = run_sensitivity_analysis(
        training,
        validation,
        concepts,
        CLASSES,
        controlled_configurations(),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "configuracoes.csv", index=False)
    unstable.to_csv(output_dir / "projetos_instaveis.csv", index=False)
    selected = select_configuration(results)
    selected.to_frame("value").to_csv(output_dir / "configuracao_selecionada.csv")
    return results, unstable, selected


if __name__ == "__main__":
    main()
