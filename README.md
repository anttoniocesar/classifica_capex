# Classificação de projetos CAPEX

Este repositório reúne o modelo exploratório de classificação de projetos
CAPEX e seus resultados.

## Documentação de negócio

- [Definição do problema, custos de erro e métricas](docs/definicao_problema_metricas.md)

## Baseline versionada

A matriz conceitual atual tem a versão `1.0`. Ao executar o modelo, o diretório
`resultados/baselines/1.0/` recebe:

- `baseline.json`, com versões e hashes das matrizes, primeira e segunda classes,
  respectivas similaridades, margem e variância explicada pelo PCA;
- `baseline_arrays.npz`, cópia compacta das matrizes `C32`, `C_EXTRA`, `P` e `V`,
  de todas as similaridades projeto versus classe e dos pesos conceituais de
  Segurança e do protótipo Hebbiano.

Os dois projetos de validação atualmente codificados pertencem a Segurança.
Seus resultados servem somente como diagnóstico e **não** como evidência de
qualidade ou acurácia do modelo.
