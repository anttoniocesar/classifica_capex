# Auditoria dos vetores de projetos

Auditoria executada sobre as três partições versionadas, com similaridade de
cosseno mínima de **0,99** para candidatos quase idênticos.

## Resumo

- projetos: **21** (19 treino, 2 validação e 0 teste);
- vetores únicos: **12**;
- frequências: dez vetores aparecem uma vez, um aparece duas vezes e um aparece
  nove vezes.

| Vetor | Frequência | Projetos |
|---|---:|---|
| V0001 | 1 | UJ-TR0259 |
| V0002 | 1 | ME-GU0047 |
| V0003 | 9 | ME-IR0057; BF-FQ0245.61; BF-FQ0246.80; BF-FA0116.80; BF-GA0126.61; BF-CO0158.80; BF-CO0144.80; RS-TR0143; RS-LA0083 |
| V0004 | 1 | ME-BH0050 |
| V0005 | 1 | UJ-EU0109 |
| V0006 | 1 | UM-GR0499 |
| V0007 | 1 | UM-GR0500 |
| V0008 | 1 | UJ-EU0118 |
| V0009 | 1 | RS-TR0141 |
| V0010 | 2 | BF-GA0117.61; BF-CO0143.61 |
| V0011 | 1 | BF-BU0149.61 |
| V0012 | 1 | UJ-EU0117 |

## Conclusão sobre as duplicatas

Os dois grupos idênticos estão **pendentes de revisão humana**. A base contém
apenas código, classe, vetor e proveniência genérica; não contém título, escopo,
local, ativo, família nem justificativa que permita concluir com segurança se
são o mesmo projeto repetido, projetos distintos com dados incompletos,
projetos equivalentes ou perda de granularidade da codificação. Portanto, não é
válido escolher arbitrariamente uma dessas quatro conclusões.

A concentração de nove projetos na mesma representação é um sinal forte a ser
investigado de preenchimento padronizado/incompleto ou granularidade
insuficiente, mas não constitui prova. A auditoria passa a exigir, para encerrar
cada grupo, `duplicate_decision` com uma das quatro conclusões e
`duplicate_justification`. Se forem investimentos distintos e válidos, a
justificativa deve explicar explicitamente por que escopos diferentes produzem
os mesmos 42 valores.

Foram encontrados ainda **21 pares quase idênticos**. Eles envolvem quatro
padrões: diferença apenas em X13 (dez pares), diferenças em X11 e X28 (dois),
diferenças em X35, X37 e X39 (oito), e diferença apenas em X13 entre ME-GU0047
e ME-BH0050 (um). São candidatos, não duplicatas automaticamente confirmadas.

## Cobertura das características

As 24 características sem variação são também nunca preenchidas (zero em todos
os projetos): **X05–X10, X14–X27, X30–X32 e X42**. Isso é compatível com o fato
de o corpus legado conter somente Segurança, mas impede avaliar a capacidade de
representar as demais categorias.

Estão presentes em todos os 21 projetos: **X01, X02, X03, X04, X13, X33, X34,
X38, X39 e X40**. **X28 e X41** aparecem em 20 de 21 projetos (95,24%). Esse é o
conjunto de características presentes em pelo menos 90% da base.

## Controle de influência em H e W

`H` agora é a média dos 12 vetores únicos. O treinamento Hebbiano de `W` também
usa uma ocorrência estável de cada vetor; assim, o grupo de nove não recebe
nove vezes o peso de uma representação unitária. A frequência e os códigos são
preservados neste relatório para auditoria, e a desduplicação matemática não é
tratada como decisão de identidade dos projetos.
