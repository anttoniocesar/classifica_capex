# Definição do problema, custos de erro e métricas

## 1. Decisão de modelagem

O problema será tratado como **multiclasse de rótulo principal**: cada projeto
deve receber exatamente uma das 13 categorias como categoria principal. Um
projeto pode atender a mais de um conceito (por exemplo, Segurança e
Manutenção), mas os demais conceitos serão registrados como **atributos
secundários**, sem se tornarem alvos adicionais do classificador nesta etapa.

Essa definição separa três aspectos que não devem ser confundidos:

1. **Alvo de treinamento:** multiclasse, com uma e somente uma categoria
   principal entre as 13 categorias.
2. **Informação complementar:** conceitos secundários podem ser multirrótulo e
   devem ser preservados para auditoria e uma eventual evolução do modelo.
3. **Decisão crítica de Segurança:** a saída multiclasse também será avaliada
   como **Segurança versus Não Segurança**, sem a necessidade de treinar um
   segundo modelo binário.

Esta decisão deve ser ratificada pelos especialistas de CAPEX, Segurança,
Manutenção e Meio Ambiente antes de congelar os rótulos de treinamento. A ata
da ratificação deve registrar participantes, data, exemplos limítrofes e
eventuais exceções à regra abaixo.

## 2. Regra para determinar a categoria principal

A categoria principal representa a **justificativa determinante para aprovar o
investimento**, e não todo benefício produzido pelo projeto. Os especialistas
devem aplicar a regra na seguinte ordem:

1. Identificar o problema que torna o investimento necessário e a evidência
   usada para aprová-lo (risco, requisito, falha, capacidade ou retorno).
2. Perguntar: **se os benefícios das outras categorias fossem retirados, o
   projeto ainda seria aprovado por esta justificativa?** A categoria cuja
   justificativa mantém a aprovação é a principal.
3. Classificar como **Segurança** quando a justificativa determinante for reduzir
   risco inaceitável a pessoas ou instalações, atender requisito obrigatório de
   segurança ou recompor barreira/controle crítico. Manutenção, modernização ou
   ganho operacional decorrentes ficam como conceitos secundários.
4. Não classificar como Segurança quando a melhoria de segurança for apenas um
   benefício acessório e o investimento continuar justificado principalmente por
   continuidade, custo, capacidade, meio ambiente ou outra categoria.
5. Se duas justificativas independentes forem suficientes para aprovar o mesmo
   escopo, solicitar aos responsáveis a decomposição em projetos/escopos. Se a
   decomposição não for possível, enviar o caso para revisão manual; o revisor
   atribui uma categoria principal e registra as demais como secundárias.

Todo rótulo deve guardar `categoria_principal`, `conceitos_secundarios`,
`justificativa`, `fonte_da_evidencia`, `revisor` e `data_da_decisao`. Divergências
entre especialistas devem ser resolvidas por consenso ou por um especialista
designado como árbitro, mantendo-se o histórico da decisão.

## 3. Matriz de custos

Os custos abaixo são **unidades relativas iniciais**, a serem convertidas ou
recalibradas pelos especialistas com dados de incidentes, horas de análise e
impacto financeiro. Eles tornam explícita a prioridade: perder um projeto de
Segurança custa mais do que gerar uma revisão ou um falso alerta.

| Evento | Consequência principal | Custo relativo inicial |
|---|---|---:|
| Projeto de Segurança classificado como outra categoria (falso negativo de Segurança) | Risco de não priorizar uma intervenção necessária, exposição de pessoas/ativos e possível descumprimento | **10** |
| Projeto de outra categoria classificado como Segurança (falso positivo de Segurança) | Distorção de portfólio, priorização ou verba e trabalho adicional de validação | **3** |
| Projeto ambíguo enviado para revisão manual | Tempo do especialista e aumento do prazo de decisão, evitando uma decisão automática insegura | **1** |
| Confusão entre duas categorias que não sejam Segurança | Relatório e alocação incorretos; impacto depende do par de categorias | **2** (padrão) |
| Classificação correta | Sem custo de erro | **0** |

O custo médio operacional deve ser acompanhado por:

```text
(10 × FN_segurança + 3 × FP_segurança + 1 × revisões
 + 2 × outros_erros) / total_de_projetos
```

Enviar um caso para revisão não é erro do modelo: é uma opção de **abstenção
controlada**. Seu custo deve, porém, entrar na avaliação para impedir que o
modelo obtenha artificialmente boa qualidade encaminhando casos demais.

## 4. Política de revisão manual

O modelo deve retornar as probabilidades das 13 categorias. Um projeto vai para
revisão quando ocorrer ao menos uma destas condições:

- a maior probabilidade ficar abaixo do limiar mínimo de confiança;
- a diferença entre as duas maiores probabilidades ficar abaixo da margem
  mínima;
- a probabilidade de Segurança cair na faixa de incerteza definida para a
  decisão binária;
- uma regra obrigatória ou informação essencial estiver ausente ou em conflito.

Os limiares não devem ser escolhidos apenas para maximizar acurácia. Devem ser
selecionados no conjunto de validação para minimizar o custo relativo da seção
3, respeitando simultaneamente a meta de recall de Segurança e a capacidade
mensal da equipe de revisão. O relatório deve sempre mostrar a **taxa e a
quantidade de revisões**, inclusive separadas por categoria prevista.

## 5. Métricas principais

Como o alvo é multiclasse e pode haver desbalanceamento, as métricas primárias
do modelo são:

1. **Recall de Segurança**, primeira restrição de aceitação, devido ao maior
   custo dos falsos negativos.
2. **Custo médio operacional**, usando a matriz de custos e incluindo revisões.
3. **F1 macro das 13 categorias**, para que classes frequentes não escondam o
   desempenho ruim das classes menores.

Devem ser apresentados como métricas de apoio: precision macro, recall macro,
F1 e suporte por classe, matriz de confusão 13 × 13, acurácia balanceada,
acurácia global e cobertura automática (percentual não enviado para revisão).
A acurácia global, isoladamente, não é critério de aprovação.

### 5.1 Visão obrigatória de Segurança

Para calcular esta visão, converter `Cat 1 - Segurança` em positivo e agregar as
outras 12 categorias como negativo. O painel e todo relatório de validação
devem incluir obrigatoriamente:

- **Recall de Segurança:** `TP / (TP + FN)`;
- **Precision de Segurança:** `TP / (TP + FP)`;
- **Falsos negativos (FN):** quantidade e taxa de projetos de Segurança que o
  fluxo automático liberou como Não Segurança;
- **Falsos positivos (FP):** quantidade e taxa de projetos Não Segurança que o
  fluxo automático classificou como Segurança;
- matriz de confusão binária completa (`TP`, `TN`, `FP`, `FN`);
- curva precision-recall e resultado no limiar operacional escolhido.

Casos enviados para revisão devem ser reportados separadamente e não podem ser
contados como acertos. Para medir o risco do fluxo antes da decisão humana,
publicar também uma avaliação conservadora em que projetos de Segurança enviados
à revisão contam como Segurança não identificada automaticamente.

## 6. Protocolo de validação e aceite

- Separar treino, validação e teste por projeto e, quando aplicável, por período,
  impedindo vazamento de versões do mesmo projeto.
- Ajustar pesos, limiares e faixa de revisão somente no conjunto de validação.
- Publicar as métricas finais uma única vez no teste, com intervalos de confiança
  quando o tamanho da amostra permitir.
- Analisar individualmente todos os FN e FP de Segurança e uma amostra dos casos
  enviados para revisão.
- Definir com os especialistas, antes do teste final, a meta mínima de recall de
  Segurança, o custo máximo aceitável e a capacidade de revisão. Sem esses três
  valores, o resultado é exploratório e não está aprovado para decisão automática.

## 7. Registro de ratificação pelos especialistas

| Campo | Preenchimento |
|---|---|
| Data da sessão | Pendente |
| Especialistas participantes | Pendente |
| Decisão sobre alvo multiclasse principal | Pendente de ratificação |
| Regra de categoria principal e exceções | Pendente de ratificação |
| Custos relativos ou financeiros aprovados | Pendente de calibração |
| Meta mínima de recall de Segurança | Pendente |
| Limite de custo médio | Pendente |
| Capacidade mensal de revisão | Pendente |
| Responsável pela aprovação | Pendente |

