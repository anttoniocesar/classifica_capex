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

Os parâmetros `minimum_similarity` e `minimum_margin` são calibrados na
partição de desenvolvimento, que obrigatoriamente deve conter projetos de
todas as classes. A calibração maximiza a cobertura automática sujeita à
precisão-alvo; ela falha explicitamente se receber apenas os 19 projetos de
Segurança. Cada decisão e cada linha exportada registra os dois limiares usados,
permitindo reproduzir e auditar o encaminhamento para revisão manual.

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

O [protocolo de curadoria e congelamento](docs/protocolo_curadoria_dataset.md)
define a coleta independente da matriz conceitual, os casos de fronteira, a
desduplicação por família e a custódia do teste. O corpus atual é legado,
exclusivamente de Segurança, e não deve ser apresentado como avaliação das 13
categorias; o teste continua vazio até haver dados reais suficientes.

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

## Auditoria de vetores repetidos

`src.dataset_audit.audit_feature_vectors` reporta o total de projetos, vetores
únicos, frequência e projetos de cada vetor, características constantes, nunca
preenchidas (zero em todo o corpus) e presentes em pelo menos 90% dos projetos.
Também lista pares com similaridade de cosseno a partir de 0,99. Esse limiar só
gera candidatos para revisão humana; similaridade não prova identidade.

Cada grupo idêntico deve receber no registro de curadoria uma das decisões
`same_project`, `different_projects_incomplete_data`, `equivalent_projects` ou
`insufficient_coding_granularity`, além de `duplicate_justification`. Projetos
distintos só são aceitos com justificativa da representação coincidente.

Para impedir peso desproporcional, `historical_prototype` calcula `H` sobre os
vetores únicos e `train_hebbian` elimina representações idênticas antes de
construir `W` por padrão. A opção `deduplicate=False` existe apenas para
reprodução explícita de resultados legados.
