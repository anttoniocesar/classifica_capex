# Classificação de projetos CAPEX

Este repositório reúne o modelo exploratório de classificação de projetos
CAPEX e seus resultados.

## Menu gráfico

Execute `python menu.py` para abrir o formulário desktop. O menu permite
classificar um projeto preenchendo as 42 características, treinar o protótipo
Hebbiano de Segurança e gerar/abrir a visualização PCA 3D. Como os dados atuais
possuem exemplos apenas de Segurança, a opção de treino não é apresentada como
um treinamento multiclasse; ela grava o artefato auditável
`resultados/prototipo_hebbiano_seguranca.npz`.

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

### Interface unitária

`src.classifier.classify_project` oferece uma interface estrita para consumo
por aplicações. Ela aceita as 42 características na ordem canônica ou como um
mapping com exatamente os nomes de `FEATURES`; campos ausentes, desconhecidos,
índices numéricos, valores fora da escala e vetores vazios são rejeitados antes
do cálculo.

```python
from src import ClassifierModel, classify_project

model = ClassifierModel(concepts, tuple(classes), version="1.0")
result = classify_project(
    project_code=code,
    features=values,
    model=model,
    thresholds={"minimum_similarity": 0.70, "minimum_margin": 0.30},
)
```

O resultado contém o código, as duas primeiras classes e similaridades, margem,
status e motivo de revisão, versão do modelo e as cinco características com
maior contribuição absoluta para o cosseno vencedor. A aprovação humana é um
registro separado, criado por `register_human_review`; ela preserva a decisão
automática original e exige a identificação do revisor, inclusive para classes
marcadas como de alto impacto no artefato do modelo.

## Análise de sensibilidade

`python run_sensitivity.py` executa uma grade controlada (um fator por vez) para
os pesos dos blocos C32 e C_EXTRA, ETA, similaridade mínima, margem mínima e,
no modo sequencial, sementes que determinam a ordem dos projetos. O relatório
registra métricas de Segurança e macro, decisões alteradas em relação à
referência, revisões manuais e projetos instáveis em
`resultados/sensibilidade/`.

A escolha é feita exclusivamente com treino e validação. O script não carrega
a partição de teste, não passa seus dados à análise e não calcula métricas
nela. Como o corpus atual de
treino e validação contém somente Segurança, as métricas macro não constituem
evidência multiclasse; o relatório serve como infraestrutura e diagnóstico até
que a curadoria forneça validação representativa das 13 categorias.

## Protótipo histórico de Segurança

O protótipo histórico direcional normaliza cada projeto antes de calcular o
centro e normaliza novamente a média resultante. Isso evita que projetos com
maior norma tenham peso adicional. `compare_historical_prototypes` também
expõe, para auditoria, a alternativa anterior (normalizar somente a média dos
vetores brutos) e o cosseno entre os dois centros.

`calculate_historical_security_similarities` usa esse centro exclusivamente na
linha de **Cat 1 - Segurança**; as referências conceituais das outras 12 classes
são preservadas. `evaluate_historical_prototype` compara essa abordagem com o
baseline conceitual no mesmo conjunto independente, sem ajustar limiares nele.
Como `data/projects_test.csv` ainda não possui casos, nenhuma métrica empírica do
protótipo é alegada por enquanto; a avaliação deve ser executada quando a
partição de teste sob custódia for preenchida.

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
