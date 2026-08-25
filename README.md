# Classificação de projetos CAPEX

Este repositório reúne o modelo exploratório de classificação de projetos
CAPEX e seus resultados.

## Documentação de negócio

- [Definição do problema, custos de erro e métricas](docs/definicao_problema_metricas.md)
- [Esquema operacional das características X01–X42](docs/esquema_caracteristicas_X01_X42.json) — inclui escala, evidências, fronteiras entre conceitos, tratamento de informação desconhecida, revisão especializada e histórico de versões

## Baseline oficial

O primeiro baseline oficial é o **classificador conceitual por similaridade de
cosseno** implementado em `src/classifier.py`. Ele compara explicitamente a
matriz de características dos projetos com a matriz conceitual das classes,
sem treinamento e sem usar o protótipo Hebbiano. A execução padrão (`main.py`)
usa esse classificador e informa, por projeto, as duas classes mais próximas,
suas similaridades, a margem e o status da decisão.

## Baseline exploratória versionada (legada)

A matriz conceitual atual tem a versão `1.0`. O fluxo exploratório legado gera,
no diretório `resultados/baselines/1.0/`:

- `baseline.json`, com versões e hashes das matrizes, primeira e segunda classes,
  respectivas similaridades, margem e variância explicada pelo PCA;
- `baseline_arrays.npz`, cópia compacta das matrizes `C32`, `C_EXTRA`, `P` e `V`,
  de todas as similaridades projeto versus classe e dos pesos conceituais de
  Segurança e do protótipo Hebbiano.

Os dois projetos de validação atualmente codificados pertencem a Segurança.
Seus resultados servem somente como diagnóstico e **não** como evidência de
qualidade ou acurácia do modelo.

## Dados de entrada

A matriz conceitual é mantida em `data/concept_matrix.csv`, com uma linha por
classe e as 42 características canônicas. Os projetos ficam separados em
`data/projects_train.csv`, `data/projects_validation.csv` e
`data/projects_test.csv`; cada registro inclui classe real e a proveniência da
codificação e da validação do rótulo.

Use `src.data.load_concept_matrices()` e `src.data.load_project_partitions()`
para carregar os arquivos. Os carregadores rejeitam esquemas diferentes de 42
características, códigos repetidos (inclusive entre partições), classes
desconhecidas, campos de proveniência vazios, valores não numéricos, infinitos,
fora da escala de 0 a 1 ou vetores inteiramente zerados.
