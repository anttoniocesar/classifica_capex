# Protocolo de curadoria e congelamento do corpus

## Situação atual

Os 21 registros versionados são um legado composto somente por projetos de
Segurança. A proveniência informa o código que os continha, mas não demonstra
que eles ficaram fora da definição ou do ajuste da matriz conceitual. Portanto,
**eles não constituem ainda o corpus independente solicitado**, e o teste final
permanece vazio e congelado. Não foram inventados rótulos ou projetos para
preencher as categorias ausentes.

## Coleta e rotulagem

Para cada investimento real, preservar o código, título/descrição originais,
período, unidade, evidência documental, categoria principal ratificada por
especialista e codificação X01–X42. Registrar ainda:

* `project_family`: identificador comum a revisões, fases, complementos e
  variantes do mesmo investimento;
* `used_for_concept_matrix`: deve ser `false`, respaldado pelo histórico da
  oficina que definiu ou ajustou a matriz;
* `boundary_cases`: zero ou mais marcadores entre
  `seguranca_manutencao`, `seguranca_renovacoes`,
  `seguranca_meio_ambiente`, `seguranca_rejeitos`,
  `seguranca_modificacoes` e `legal_nao_seguranca`;
* justificativa da classe principal, conceitos secundários, fonte, revisor e
  data da decisão.

Um requisito legal é atributo, não regra automática de Segurança. O marcador
`legal_nao_seguranca` exige X03 ou X33 e categoria principal diferente de
Segurança, com justificativa determinante documentada.

## Desduplicação antes da divisão

Revisar em conjunto códigos normalizados semelhantes, títulos semelhantes e
vetores X01–X42 idênticos. Consolidar duplicatas verdadeiras em um registro;
quando fases devam permanecer, atribuir a mesma `project_family`. A família é a
unidade indivisível de amostragem e nunca pode atravessar partições. Vetores
idênticos são apenas candidatos à revisão, pois projetos reais distintos podem
ter a mesma codificação.

## Divisão e uso

Só dividir quando as 13 categorias tiverem suporte suficiente para treino,
validação e teste, preservando também os seis grupos de fronteira. Fazer a
divisão estratificada por categoria **por família**, não por linha:

1. desenvolvimento/treino: ajuste de modelo;
2. validação: escolha de pesos, limiares e capacidade de revisão;
3. teste final: uma única avaliação após registrar todas as decisões.

Antes do desenvolvimento, gerar e aprovar um manifesto do teste contendo os
identificadores de família e o SHA-256 do arquivo. O responsável pela custódia
não deve disponibilizar rótulos do teste à equipe de desenvolvimento. Qualquer
alteração invalida a avaliação: criar uma nova versão do corpus, justificar a
mudança e repetir o congelamento, sem sobrescrever o manifesto anterior.

`src.dataset_audit.audit_dataset` bloqueia a prontidão quando falta registro de
independência, categoria, caso de fronteira, família, ou quando uma família (ou
um vetor idêntico que precisa de revisão) atravessa partições.
