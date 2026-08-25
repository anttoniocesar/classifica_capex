from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons
from sklearn.decomposition import PCA
from scipy.stats import chi2

from baseline_modelo import save_baseline

# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

# Se você já possui funções aqui, mantenha-as.


def plot_confidence_ellipsoid_3d(
    ax,
    points,
    confidence=0.95,
    color="yellow",
    alpha=0.18,
    edgecolor="goldenrod"
):
    """
    Cria uma elipsoide 3D baseada na média
    e covariância dos pontos.
    """

    points = np.asarray(points)

    # Centro da nuvem
    center = points.mean(axis=0)

    # Matriz de covariância
    cov = np.cov(
        points,
        rowvar=False
    )

    # Regularização numérica
    cov = cov + np.eye(3) * 1e-8

    # Autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ordenar do maior para o menor
    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Escala da região
    scale = np.sqrt(
        chi2.ppf(
            confidence,
            df=3
        )
    )

    # Raios
    radii = scale * np.sqrt(eigenvalues)

    # Malha esférica
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)

    x = np.outer(
        np.cos(u),
        np.sin(v)
    )

    y = np.outer(
        np.sin(u),
        np.sin(v)
    )

    z = np.outer(
        np.ones_like(u),
        np.cos(v)
    )

    sphere = np.stack(
        [x, y, z],
        axis=-1
    )

    # Transformar esfera em elipsoide
    ellipsoid = sphere * radii

    # Rotacionar
    ellipsoid = ellipsoid @ eigenvectors.T

    # Deslocar para o centro
    ellipsoid += center

    # Superfície
    ax.plot_surface(
        ellipsoid[:, :, 0],
        ellipsoid[:, :, 1],
        ellipsoid[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=True
    )

    # Linhas da superfície
    ax.plot_wireframe(
        ellipsoid[:, :, 0],
        ellipsoid[:, :, 1],
        ellipsoid[:, :, 2],
        color=edgecolor,
        linewidth=0.3,
        alpha=0.20
    )

    return center, radii

# ============================================================
# CONFIGURAÇÕES
# ============================================================

OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)

ETA = 0.05
concept_matrix_version = "1.0"
MATRIX_VERSIONS = {
    "C32": "1.0",
    "C_EXTRA": "1.0",
    "P": "1.0",
    "V": "1.0",
}


# ============================================================
# 1. NOMES DAS 42 CARACTERÍSTICAS
# ============================================================

FEATURES = [
    "X01 Segurança ocupacional",
    "X02 Segurança industrial",
    "X03 Requisito legal",
    "X04 Gestão de riscos",
    "X05 Aumento de capacidade",
    "X06 Aumento de produção",
    "X07 Novo produto",
    "X08 Novo mercado",
    "X09 Redução de custos",
    "X10 Melhoria da qualidade",
    "X11 Substituição / renovação",
    "X12 Obsolescência",
    "X13 Continuidade operacional",
    "X14 Grande reforma / reconstrução",
    "X15 Meio ambiente",
    "X16 Efluentes / resíduos",
    "X17 Tecnologia da informação",
    "X18 Pesquisa e desenvolvimento",
    "X19 Estudo / piloto / protótipo",
    "X20 SPA / obrigação contratual",
    "X21 Cilindros de laminação",
    "X22 Redução de CO2",
    "X23 Mudança de combustível",
    "X24 Rejeitos",
    "X25 Barragem / armazenamento rejeitos",
    "X26 Retorno financeiro",
    "X27 Nova instalação / greenfield",
    "X28 Modificação ativo existente",
    "X29 Melhoria tecnológica normal",
    "X30 Incerteza tecnológica",
    "X31 Redução capital de giro",
    "X32 Capitalização / monitoramento especial",
    "X33 Adequação normativa",
    "X34 Proteção de pessoas / operadores",
    "X35 Proteção de máquinas / equipamentos",
    "X36 Proteção contra incêndio",
    "X37 Segurança elétrica / SPDA",
    "X38 Exposição direta de pessoas ao risco",
    "X39 Deficiência de barreiras / controles críticos",
    "X40 Condição crítica / desvio do processo",
    "X41 Falta de sensor / intertravamento / redundância",
    "X42 Automação para redução da exposição humana",
]


# ============================================================
# 2. NOMES DAS 13 CLASSES
# ============================================================

CLASSES = [
    "Cat 1 - Segurança",
    "Cat 2 - Crescimento",
    "Cat 3 - Modificações",
    "Cat 4 - Manutenção",
    "Cat 5 - Renovações",
    "Cat 6 - Meio ambiente",
    "Cat 7 - TI",
    "Cat 8 - P&D",
    "Cat 9 - SPA Crescimento",
    "Cat 10 - SPA Manut./Amb.",
    "Cat 11 - Cilindros",
    "Cat 12 - Descarbonização",
    "Cat 13 - Rejeitos",
]


# ============================================================
# 3. MATRIZ CONCEITUAL ORIGINAL 13 x 32
# ============================================================

C32 = np.array([
    # Cat 1 - Segurança
    [
        1, 1, .60, 1, 0, 0, 0, 0,
        0, 0, .25, .75, .50, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, .50, .25, 0, 0, 0
    ],

    # Cat 2 - Crescimento
    [
        0, 0, 0, 0, 1, 1, .75, .75,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, .75, 1, .50, 0, 0, 0, 0
    ],

    # Cat 3 - Modificações
    [
        0, 0, 0, 0, .50, .25, 0, 0,
        1, 1, 0, 0, .25, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, .75, 0, 1, .50, 0, .75, 0
    ],

    # Cat 4 - Manutenção
    [
        0, 0, 0, .25, 0, 0, 0, 0,
        .25, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, .50, .75, 0, 0, 0
    ],

    # Cat 5 - Renovações
    [
        0, 0, 0, .25, 0, 0, 0, 0,
        0, .25, .75, .75, .50, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, .75, .75, 0, 0, 0
    ],

    # Cat 6 - Meio ambiente
    [
        0, 0, .75, .25, 0, 0, 0, 0,
        0, 0, 0, 0, .25, 0, 1, 1,
        0, 0, 0, 0, 0, .50, .25, .25,
        0, 0, 0, .50, .25, 0, 0, 0
    ],

    # Cat 7 - TI
    [
        0, 0, 0, .25, 0, 0, 0, 0,
        0, 0, .50, .50, .50, 0, 0, 0,
        1, .25, 0, 0, 0, 0, 0, 0,
        0, 0, 0, .50, .75, .25, 0, 0
    ],

    # Cat 8 - P&D
    [
        0, 0, 0, 0, 0, 0, .50, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, .50, 1, 0, 0
    ],

    # Cat 9 - SPA Crescimento
    [
        0, 0, .50, 0, .75, .75, .50, .50,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, .50, .75, .50, 0, 0, 0, 0
    ],

    # Cat 10 - SPA Manutenção / Ambiente
    [
        0, 0, .75, .25, 0, 0, 0, 0,
        0, 0, .75, .50, .75, 0, .75, .50,
        0, 0, 0, 1, 0, 0, 0, .25,
        0, 0, 0, .50, .50, 0, 0, 0
    ],

    # Cat 11 - Cilindros
    [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, .25, 0, .25, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1
    ],

    # Cat 12 - Descarbonização
    [
        0, 0, .25, .25, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, .75, .25,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, .50, .50, 0, 0, 0
    ],

    # Cat 13 - Rejeitos
    [
        0, .25, .50, .75, 0, 0, 0, 0,
        0, 0, .50, .25, .75, .25, .50, .25,
        0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, .50, 0, 0, 0, 0
    ]
], dtype=float)


# ============================================================
# 4. EXTENSÃO X33:X42
# ============================================================

C_EXTRA = np.array([
    # X33 X34 X35 X36 X37 X38 X39 X40 X41 X42

    [0.75, 1.00, 0.90, 0.75, 0.75, 1.00, 1.00, 1.00, 0.90, 0.75],  # Segurança
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Crescimento
    [0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25, 0.25],  # Modificações
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.00],  # Manutenção
    [0.00, 0.00, 0.25, 0.25, 0.25, 0.00, 0.00, 0.25, 0.25, 0.00],  # Renovações
    [0.50, 0.25, 0.00, 0.00, 0.00, 0.25, 0.25, 0.25, 0.25, 0.00],  # Meio ambiente
    [0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # TI
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25, 0.50],  # P&D
    [0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # SPA Cresc.
    [0.50, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.00],  # SPA Manut/Amb
    [0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25, 0.00],  # Cilindros
    [0.25, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25, 0.25],  # Descarbonização
    [0.50, 0.50, 0.25, 0.00, 0.00, 0.50, 0.50, 0.50, 0.25, 0.00],  # Rejeitos
], dtype=float)

C = np.hstack([C32, C_EXTRA])

assert C.shape == (13, 42)

def normalize_rows(matrix):
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms != 0
    )

# ============================================================
# FASE 1 — VALIDAÇÃO CONCEITUAL
# ============================================================
#
# Projetos novos usados exclusivamente para testar
# os 13 conceitos.
#
# IMPORTANTE:
# Estes projetos NÃO participam do treinamento Hebbiano.
# ============================================================

VALIDATION_CODES = [

    # CAT 1 — SEGURANÇA
    "UJ-EU0117",
    "RS-LA0083",
    #"UJ-MS0049",
    #"UP-MS0030",
    #"FT-TT0181",

    # CAT 2 — CRESCIMENTO
    #"BM-LA0086",
    #"MS-GR0099",

    # CAT 3 — MODIFICAÇÕES
    #"FT-SP0006",
    #"FT-SP0005",
    #"RS-TR0150",
    #"BM-GT0011",
    #"UP-AC0248",

    # CAT 4 — MANUTENÇÃO
    #"US-SA0019",
    #"RP-RP0001",
    #"RS-LA0078",
    #"FT-SP0007",
    #"UJ-AF0118",

    # CAT 5 — RENOVAÇÕES
    #"UJ-AF0019",

    # CAT 6 — MEIO AMBIENTE
    #"BM-MA0010",
    #"US-SA0002",
    #"ME-CT0019",
    #"UJ-MA0039",
    #"UJ-TR0262",

    # CAT 7 — TI
    #"CD-TI0114",
    #"CD-TI0115",
    #"UP-TI0051",
    #"CD-TI0126",
    #"FT-TI0034",

    # CAT 8 — P&D
    #"FT-SP0002",
]


print("\nProjetos de validação:")
print(len(VALIDATION_CODES))

VALIDATION_REAL_CLASSES = [
    "Cat 1 - Segurança",
    "Cat 1 - Segurança",
    #"Cat 1 - Segurança",
    #"Cat 1 - Segurança",
    #"Cat 1 - Segurança",

    #"Cat 2 - Crescimento",
    #"Cat 2 - Crescimento",

    #"Cat 3 - Modificações",
    #"Cat 3 - Modificações",
    #"Cat 3 - Modificações",
    #"Cat 3 - Modificações",
    #"Cat 3 - Modificações",

    #"Cat 4 - Manutenção",
    #"Cat 4 - Manutenção",
    #"Cat 4 - Manutenção",
    #"Cat 4 - Manutenção",
    #"Cat 4 - Manutenção",

    #"Cat 5 - Renovações",

    #"Cat 6 - Meio ambiente",
    #"Cat 6 - Meio ambiente",
    #"Cat 6 - Meio ambiente",
    #"Cat 6 - Meio ambiente",
    #"Cat 6 - Meio ambiente",

    #"Cat 7 - TI",
    #"Cat 7 - TI",
    #"Cat 7 - TI",
    #"Cat 7 - TI",
    #"Cat 7 - TI",

    #"Cat 8 - P&D",
]

assert len(VALIDATION_CODES) == len(VALIDATION_REAL_CLASSES)

# ============================================================
# MATRIZ DOS PROJETOS DE VALIDAÇÃO
# ============================================================

V = np.zeros(
    (
        len(VALIDATION_CODES),
        42
    ),
    dtype=float
)


def set_validation_values(row, values):
    """
    Preenche as características X01...X42
    de um projeto da base de validação.
    """

    for feature_number, value in values.items():

        V[
            row,
            feature_number - 1
        ] = value


# ============================================================
# PASSO 4 — CODIFICAR OS NOVOS PROJETOS
# ============================================================

# Projeto 0 — UJ-EU0117 — AVCB
set_validation_values(
    0,
    {
        1: 0.50,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        13: 0.50,
        28: 0.50,
        33: 1.00,
        34: 0.75,
        36: 1.00,
        38: 0.75,
        39: 0.75,
        40: 0.75
    }
)


# Projeto 1 — RS-LA0083 — NR12
set_validation_values(
    1,
    {
        1: 1.00,
        2: 0.75,
        3: 1.00,
        4: 1.00,
        13: 0.50,
        28: 0.75,
        33: 1.00,
        34: 1.00,
        35: 1.00,
        38: 1.00,
        39: 1.00,
        40: 0.50,
        41: 0.75
    }
)

print("\nVALIDAÇÃO DA MATRIZ V")
print("Formato da matriz V:", V.shape)

empty_validation_rows = np.where(
    np.linalg.norm(V, axis=1) == 0
)[0]

print(
    "Projetos ainda não codificados:",
    empty_validation_rows
) 

if len(empty_validation_rows) > 0:
    raise ValueError(
        f"Existem projetos de validação ainda não codificados: "
        f"{empty_validation_rows}"
    )

# ============================================================
# PASSO 5 — SIMILARIDADE CONCEITUAL
# ============================================================

V_norm = normalize_rows(V)

C_validation_norm = normalize_rows(C)

validation_similarities = V_norm @ C_validation_norm.T

# ============================================================
# PASSO 6 — ENCONTRAR PRIMEIRA E SEGUNDA CLASSES
# ============================================================

validation_ranking = np.argsort(
    validation_similarities,
    axis=1
)[:, ::-1]

validation_winner_index = (
    validation_ranking[:, 0]
)

validation_second_index = (
    validation_ranking[:, 1]
)

validation_winner_score = (
    validation_similarities[
        np.arange(len(V)),
        validation_winner_index
    ]
)

validation_second_score = (
    validation_similarities[
        np.arange(len(V)),
        validation_second_index
    ]
)

# ============================================================
# PASSO 7 — CALCULAR A MARGEM ENTRE A 1ª E A 2ª CLASSE
# ============================================================

validation_margin = (
    validation_winner_score
    -
    validation_second_score
)

# ============================================================
# PASSO 8 — DIAGNÓSTICO EXPLORATÓRIO
# ============================================================

# Converte o índice da classe vencedora
# para o nome da classe
validation_predicted_classes = [
    CLASSES[i]
    for i in validation_winner_index
]


print("\n" + "=" * 70)
print("PASSO 8 — DIAGNÓSTICO (NÃO É EVIDÊNCIA DE QUALIDADE)")
print("=" * 70)

for code, real, predicted in zip(
    VALIDATION_CODES,
    VALIDATION_REAL_CLASSES,
    validation_predicted_classes,
):
    print(
        f"{code:15s} | "
        f"Real: {real:25s} | "
        f"Prevista: {predicted:25s}"
    )

print("Os dois projetos são de Segurança; não se calcula acurácia como qualidade.")

# ============================================================
# 5. PROJETOS REAIS DE SEGURANÇA
# ============================================================

PROJECT_CODES = [
    "UJ-TR0259",
    "ME-GU0047",
    "ME-IR0057",
    "ME-BH0050",
    "UJ-EU0109",
    "BF-FQ0245.61",
    "BF-FQ0246.80",
    "BF-FA0116.80",
    "BF-GA0126.61",
    "BF-CO0158.80",
    "UM-GR0499",
    "UM-GR0500",
    "UJ-EU0118",
    "RS-TR0141",
    "BF-GA0117.61",
    "BF-BU0149.61",
    "BF-CO0143.61",
    "BF-CO0144.80",
    "RS-TR0143",
]

P = np.zeros((19, 42), dtype=float)


def set_values(row, values):
    for feature_number, value in values.items():
        P[row, feature_number - 1] = value


# ============================================================
# 6. CODIFICAÇÃO DOS 19 PROJETOS
# ============================================================

set_values(0, {
    1: 1, 2: .75, 3: 1, 4: 1, 13: .25, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(1, {
    1: .50, 2: 1, 3: .75, 4: 1,
    11: .75, 12: 1, 13: .75, 28: .75, 29: .75,
    33: .75, 34: .75, 36: 1,
    38: .75, 39: .75, 40: 1, 41: .75
})

set_values(2, {
    1: 1, 2: .75, 3: 1, 4: 1, 13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(3, {
    1: .50, 2: 1, 3: .75, 4: 1,
    11: .75, 12: 1, 13: .50, 28: .75, 29: .75,
    33: .75, 34: .75, 36: 1,
    38: .75, 39: .75, 40: 1, 41: .75
})

set_values(4, {
    1: 1, 2: .75, 3: 1, 4: 1,
    11: .25, 13: .50, 28: .50,
    33: 1, 34: 1, 37: 1,
    38: 1, 39: .75, 40: .50, 41: .50
})

set_values(5, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(6, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(7, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(8, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(9, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(10, {
    1: .50, 2: 1, 3: 1, 4: 1,
    11: .50, 12: .75, 13: .75, 28: .75,
    33: 1, 34: .75, 36: 1,
    38: .75, 39: .75, 40: 1, 41: .75
})

set_values(11, {
    1: 1, 2: .75, 3: 1, 4: 1,
    11: .50, 12: .50, 13: .75, 28: .75,
    33: 1, 34: 1, 37: 1,
    38: 1, 39: .75, 40: .50, 41: .50
})

set_values(12, {
    1: 1, 2: .75, 3: 1, 4: .75,
    13: .50,
    33: 1, 34: 1, 35: .50,
    38: .75, 39: .50, 40: .75, 41: .50
})

set_values(13, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .50,
    33: 1, 34: 1,
    38: 1, 39: 1, 40: .25, 41: .25
})

set_values(14, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 37: 1,
    38: 1, 39: .75, 40: .50, 41: .50
})

set_values(15, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: .75, 37: .25,
    38: 1, 39: .75, 40: .50, 41: .75
})

set_values(16, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 37: 1,
    38: 1, 39: .75, 40: .50, 41: .50
})

set_values(17, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})

set_values(18, {
    1: 1, 2: .75, 3: 1, 4: 1,
    13: .50, 28: .75,
    33: 1, 34: 1, 35: 1,
    38: 1, 39: 1, 40: .50, 41: .75
})


# ============================================================
# 7. FUNÇÕES AUXILIARES
# ============================================================

def normalize_vector(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector.copy()

    return vector / norm


def interpretar_similaridade(sim):
    """
    Interpretação didática da similaridade de cosseno.
    """

    if sim >= 0.90:
        return "Muito próximos"

    elif sim >= 0.75:
        return "Relativamente próximos"

    elif sim >= 0.50:
        return "Diferença importante"

    elif sim >= 0.25:
        return "Bastante diferentes"

    else:
        return "Muito distintos"


# ============================================================
# 8. SIMILARIDADE PROJETO x CLASSE
# ============================================================

P_norm = normalize_rows(P)
C_norm = normalize_rows(C)

# ============================================================
# SIMILARIDADE DE COSSENO ENTRE AS 13 CLASSES CONCEITUAIS
# ============================================================

# Cada linha de C_norm representa uma classe normalizada.
#
# A multiplicação:
#
#     C_norm @ C_norm.T
#
# compara cada classe com todas as outras classes.
#
# O resultado será uma matriz 13 x 13.

concept_similarity = C_norm @ C_norm.T


# Transformar a matriz NumPy em DataFrame
# para facilitar visualização e exportação para Excel.

concept_similarity_df = pd.DataFrame(
    concept_similarity,
    index=CLASSES,
    columns=CLASSES
)


# ============================================================
# DISTÂNCIA DE COSSENO ENTRE AS 13 CLASSES
# ============================================================

# Distância de cosseno:
#
#     distância = 1 - similaridade
#
# Exemplo:
#
# similaridade = 0.90
# distância    = 0.10

concept_cosine_distance = 1 - concept_similarity


concept_cosine_distance_df = pd.DataFrame(
    concept_cosine_distance,
    index=CLASSES,
    columns=CLASSES
)

# ============================================================
# PARES DE CLASSES — SIMILARIDADE E DISTÂNCIA
# ============================================================

pairs = []

for i in range(len(CLASSES)):

    for j in range(i + 1, len(CLASSES)):

        sim = concept_similarity[i, j]

        distance = 1 - sim

        pairs.append({
            "Classe A": CLASSES[i],
            "Classe B": CLASSES[j],
            "Similaridade Cosseno": sim,
            "Distância Cosseno": distance,
            "Interpretação": interpretar_similaridade(sim)
        })


concept_pairs_df = pd.DataFrame(pairs)

concept_pairs_df = concept_pairs_df.sort_values(
    "Similaridade Cosseno",
    ascending=False
)

similarities = P_norm @ C_norm.T

similarity_df = pd.DataFrame(
    similarities,
    index=PROJECT_CODES,
    columns=CLASSES
)

ranking = np.argsort(similarities, axis=1)[:, ::-1]

winner_index = ranking[:, 0]
second_index = ranking[:, 1]

winner_scores = similarities[
    np.arange(len(P)),
    winner_index
]

second_scores = similarities[
    np.arange(len(P)),
    second_index
]

ranking_df = pd.DataFrame({
    "Projeto": PROJECT_CODES,
    "Classe prevista": [CLASSES[i] for i in winner_index],
    "Similaridade 1": winner_scores,
    "2ª classe": [CLASSES[i] for i in second_index],
    "Similaridade 2": second_scores,
    "Margem": winner_scores - second_scores
})


# ============================================================
# 9. PROTÓTIPO HISTÓRICO DA SEGURANÇA
# ============================================================

H = P.mean(axis=0)
H_norm = normalize_vector(H)

C_security = C[0]
C_security_norm = normalize_vector(C_security)

sim_concept_history = float(
    np.dot(C_security_norm, H_norm)
)


# ============================================================
# 10. TREINAMENTO HEBBIANO
# ============================================================

W = C_security_norm.copy()

hebb_history = [W.copy()]

for project in P:
    project_norm = normalize_vector(project)

    W = W + ETA * project_norm
    W = normalize_vector(W)

    hebb_history.append(W.copy())


sim_hebb_concept = float(
    np.dot(W, C_security_norm)
)

sim_hebb_history = float(
    np.dot(W, H_norm)
)


# ============================================================
# 11. TABELA DE PESOS
# ============================================================

weights_df = pd.DataFrame({
    "Característica": FEATURES,
    "Conceito bruto": C_security,
    "Conceito normalizado": C_security_norm,
    "Histórico bruto": H,
    "Histórico normalizado": H_norm,
    "Hebb": W
})

weights_df["Delta Hebb-Conceito"] = (
    weights_df["Hebb"]
    - weights_df["Conceito normalizado"]
)


# ============================================================
# 12. PCA CORRIGIDO — ESPAÇO NORMALIZADO
# ============================================================
#
# IMPORTANTE:
# A classificação usa similaridade de cosseno.
# Portanto, o que importa é a DIREÇÃO dos vetores,
# e não sua magnitude.
#
# Para que o PCA represente a mesma lógica geométrica,
# normalizamos:
#
#   - 13 conceitos
#   - 19 projetos
#   - centro histórico
#   - protótipo Hebbiano
#
# antes da projeção.
# ============================================================


# ------------------------------------------------------------
# 12.1 Normalização
# ------------------------------------------------------------

C_pca_input = normalize_rows(C)
P_pca_input = normalize_rows(P)

H_pca_input = normalize_vector(H)

# W já foi normalizado durante o treinamento,
# mas normalizamos novamente por segurança.
W_pca_input = normalize_vector(W)


# ------------------------------------------------------------
# 12.2 Ajustar PCA SOMENTE nos 13 conceitos normalizados
# ------------------------------------------------------------

pca = PCA(n_components=3)

C_pca = pca.fit_transform(
    C_pca_input
)


# ------------------------------------------------------------
# 12.3 Projetar os projetos no MESMO espaço PCA
# ------------------------------------------------------------

P_pca = pca.transform(
    P_pca_input
)


# ------------------------------------------------------------
# 12.4 Projetar o centro histórico normalizado
# ------------------------------------------------------------

H_pca = pca.transform(
    H_pca_input.reshape(1, -1)
)[0]


# ------------------------------------------------------------
# 12.5 Projetar o protótipo Hebbiano
# ------------------------------------------------------------

W_pca = pca.transform(
    W_pca_input.reshape(1, -1)
)[0]


# ------------------------------------------------------------
# 12.6 Variância explicada
# ------------------------------------------------------------

variance = (
    pca.explained_variance_ratio_
    * 100
)


print("\n" + "=" * 70)
print("PCA CORRIGIDO — VETORES NORMALIZADOS")
print("=" * 70)

print(
    f"\nPC1 = {variance[0]:.4f}%"
)

print(
    f"PC2 = {variance[1]:.4f}%"
)

print(
    f"PC3 = {variance[2]:.4f}%"
)

print(
    f"TOTAL = {variance.sum():.4f}%"
)

print(
    "\nH_pca normalizado:"
)

print(
    np.round(H_pca, 4)
)

baseline_metadata_path, baseline_arrays_path = save_baseline(
    OUTPUT_DIR,
    concept_matrix_version=concept_matrix_version,
    matrix_versions=MATRIX_VERSIONS,
    classes=CLASSES,
    project_codes=PROJECT_CODES,
    validation_codes=VALIDATION_CODES,
    c32=C32,
    c_extra=C_EXTRA,
    projects=P,
    validation_projects=V,
    project_similarities=similarities,
    validation_similarities=validation_similarities,
    conceptual_security_weights=C_security,
    normalized_conceptual_security_weights=C_security_norm,
    hebbian_weights=W,
    pca_explained_variance=variance,
)

print(
    "\nW_pca normalizado:"
)

print(
    np.round(W_pca, 4)
)


# ============================================================
# 13. TABELAS DE COORDENADAS PCA
# ============================================================

classes_pca_df = pd.DataFrame(
    C_pca,
    index=CLASSES,
    columns=["PC1", "PC2", "PC3"]
)

projects_pca_df = pd.DataFrame(
    P_pca,
    index=PROJECT_CODES,
    columns=["PC1", "PC2", "PC3"]
)

summary_pca_df = pd.DataFrame({
    "Objeto": [
        "Centro histórico Segurança",
        "Protótipo Hebbiano Segurança"
    ],
    "PC1": [
        H_pca[0],
        W_pca[0]
    ],
    "PC2": [
        H_pca[1],
        W_pca[1]
    ],
    "PC3": [
        H_pca[2],
        W_pca[2]
    ]
})


# ============================================================
# 14. GRÁFICO 3D
# ============================================================

fig = plt.figure(
    figsize=(18, 12)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

colors = plt.cm.tab20(
    np.linspace(0, 1, len(CLASSES))
)

# ============================================================
# CLASSE DE REFERÊNCIA PARA AS SIMILARIDADES NO GRÁFICO
# ============================================================

# Cat 1 - Segurança é a posição 0 da matriz
security_index = 0

# Similaridade de cosseno de Segurança com todas as classes
security_similarities = (
    concept_similarity[security_index]
)

# Distância de cosseno:
# distância = 1 - similaridade
security_distances = (
    concept_cosine_distance[security_index]
)

# ============================================================
# CLASSES CONCEITUAIS
# ============================================================

for i, class_name in enumerate(CLASSES):

    x, y, z = C_pca[i]

    # Desenha a estrela da classe
    ax.scatter(
        x,
        y,
        z,
        s=180,
        marker="*",
        color=colors[i],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95
    )

    # --------------------------------------------------------
    # Texto mostrado ao lado da estrela
    # --------------------------------------------------------

    if i == security_index:

        # A própria Segurança é nossa referência
        label = (
            f"  {class_name}\n"
            f"  Referência"
        )

    else:

        # Demais classes:
        # mostra similaridade e distância em relação à Segurança
        label = (
            f"  {class_name}\n"
            f"  Sim={security_similarities[i]:.2f}"
        )

    ax.text(
        x,
        y,
        z,
        label,
        fontsize=8
    )


# ============================================================
# PROJETOS REAIS DE SEGURANÇA
# ============================================================

scatter_projects = ax.scatter(
    P_pca[:, 0],
    P_pca[:, 1],
    P_pca[:, 2],
    s=70,
    color="#1565C0",
    alpha=0.72,
    edgecolor="white",
    linewidth=0.5,
    label="19 projetos reais de Segurança"
)

# ============================================================
# REGIÃO DE DISPERSÃO DOS PROJETOS DE SEGURANÇA
# ============================================================

ellipsoid_center, ellipsoid_radii = (
    plot_confidence_ellipsoid_3d(
        ax,
        P_pca,
        confidence=0.95,
        color="yellow",
        alpha=0.18
    )
)

# ============================================================
# CENTRO HISTÓRICO
# ============================================================

scatter_history = ax.scatter(
    H_pca[0],
    H_pca[1],
    H_pca[2],
    color="black",
    marker="o",
    s=220,
    edgecolor="white",
    linewidth=1.5,
    label="Centro histórico Segurança"
)


# ============================================================
# PROTÓTIPO HEBBIANO
# ============================================================

scatter_hebb = ax.scatter(
    W_pca[0],
    W_pca[1],
    W_pca[2],
    color="red",
    marker="X",
    s=260,
    edgecolor="black",
    linewidth=1.0,
    label="Protótipo Hebbiano Segurança"
)


# Vetor conceito -> Hebb
security_coord = C_pca[0]

ax.plot(
    [
        security_coord[0],
        W_pca[0]
    ],
    [
        security_coord[1],
        W_pca[1]
    ],
    [
        security_coord[2],
        W_pca[2]
    ],
    color="red",
    linestyle="--",
    linewidth=2.5,
    alpha=0.9
)


# Vetor conceito -> histórico
ax.plot(
    [
        security_coord[0],
        H_pca[0]
    ],
    [
        security_coord[1],
        H_pca[1]
    ],
    [
        security_coord[2],
        H_pca[2]
    ],
    color="black",
    linestyle=":",
    linewidth=2.0,
    alpha=0.7
)


ax.set_title(
    "PCA 3D — 13 Classes Conceituais + 19 Projetos Reais de Segurança",
    fontsize=18,
    fontweight="bold",
    pad=24
)

ax.set_xlabel(
    f"PC1 ({variance[0]:.2f}%)",
    fontsize=11
)

ax.set_ylabel(
    f"PC2 ({variance[1]:.2f}%)",
    fontsize=11
)

ax.set_zlabel(
    f"PC3 ({variance[2]:.2f}%)",
    fontsize=11
)

ax.view_init(
    elev=24,
    azim=-58
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)

plt.tight_layout()

graph_path = (
    OUTPUT_DIR /
    "pca_3d_seguranca_atualizado.png"
)

plt.savefig(
    graph_path,
    dpi=220,
    bbox_inches="tight"
)

# ============================================================
# CONTROLE INTERATIVO — PROJETOS REAIS
# ============================================================

ax_check = plt.axes([
    0.79,   # posição horizontal
    0.75,   # posição vertical
    0.19,   # largura
    0.15    # altura
])

check_elements = CheckButtons(
    ax_check,
    [
        "Projetos reais",
        "Centro histórico",
        "Protótipo Hebbiano"
    ],
    [
        True,
        True,
        True
    ]
)


def toggle_elements(label):

    if label == "Projetos reais":

        scatter_projects.set_visible(
            not scatter_projects.get_visible()
        )

    elif label == "Centro histórico":

        scatter_history.set_visible(
            not scatter_history.get_visible()
        )
    elif label == "Protótipo Hebbiano":

        scatter_hebb.set_visible(
            not scatter_hebb.get_visible()
        )

    fig.canvas.draw_idle()


check_elements.on_clicked(
    toggle_elements
)

plt.show()


# ============================================================
# 15. EXPORTAR EXCEL
# ============================================================

excel_path = (
    OUTPUT_DIR /
    "resultados_modelo_seguranca.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    pd.DataFrame(
        C,
        index=CLASSES,
        columns=FEATURES
    ).to_excel(
        writer,
        sheet_name="Matriz_Classes"
    )

    pd.DataFrame(
        P,
        index=PROJECT_CODES,
        columns=FEATURES
    ).to_excel(
        writer,
        sheet_name="Projetos_Seguranca"
    )

    similarity_df.to_excel(
        writer,
        sheet_name="Similaridades"
    )

    ranking_df.to_excel(
        writer,
        sheet_name="Ranking",
        index=False
    )

    weights_df.to_excel(
        writer,
        sheet_name="Pesos_Seguranca",
        index=False
    )

    classes_pca_df.to_excel(
        writer,
        sheet_name="PCA_Classes"
    )

    projects_pca_df.to_excel(
        writer,
        sheet_name="PCA_Projetos"
    )

    summary_pca_df.to_excel(
        writer,
        sheet_name="PCA_Resumo",
        index=False
    )

    concept_similarity_df.to_excel(
        writer,
        sheet_name="Cosine_Sim_Classes"
    )

    concept_cosine_distance_df.to_excel(
        writer,
        sheet_name="Cosine_Dist_Classes"
    )

    concept_pairs_df.to_excel(
        writer,
        sheet_name="Pares_Classes",
        index=False
    )

    pd.DataFrame({
        "Indicador": [
            "PC1 (%)",
            "PC2 (%)",
            "PC3 (%)",
            "Total 3 PCs (%)",
            "Similaridade Conceito x Histórico",
            "Similaridade Hebb x Conceito",
            "Similaridade Hebb x Histórico",
            "Taxa de aprendizagem eta"
        ],
        "Valor": [
            variance[0],
            variance[1],
            variance[2],
            variance.sum(),
            sim_concept_history,
            sim_hebb_concept,
            sim_hebb_history,
            ETA
        ]
    }).to_excel(
        writer,
        sheet_name="Resumo",
        index=False
    )


# ============================================================
# 16. SAÍDA NO TERMINAL
# ============================================================

print("\n" + "=" * 70)
print("MODELO HEBBIANO + PCA — SEGURANÇA")
print("=" * 70)

print("\nFormato matriz classes:")
print(C.shape)

print("\nFormato matriz projetos:")
print(P.shape)

print("\nVARIÂNCIA EXPLICADA PELO PCA")
print(f"PC1: {variance[0]:.4f}%")
print(f"PC2: {variance[1]:.4f}%")
print(f"PC3: {variance[2]:.4f}%")
print(f"TOTAL: {variance.sum():.4f}%")

print("\nC_pca:")
print(
    classes_pca_df
    .round(4)
    .to_string()
)

print("\nP_pca:")
print(
    projects_pca_df
    .round(4)
    .to_string()
)

print("\nH_pca:")
print(
    np.round(
        H_pca,
        4
    )
)

print("\nW_pca:")
print(
    np.round(
        W_pca,
        4
    )
)

print("\nSIMILARIDADES GERAIS")
print(
    f"Conceito Segurança x Histórico: "
    f"{sim_concept_history:.4f}"
)

print(
    f"Hebb x Conceito: "
    f"{sim_hebb_concept:.4f}"
)

print(
    f"Hebb x Histórico: "
    f"{sim_hebb_history:.4f}"
)

print("\nRANKING PROJETO x CLASSE")
print(
    ranking_df
    .round(4)
    .to_string(index=False)
)

print("\n15 MAIORES PESOS HEBBIANOS")
print(
    weights_df
    .sort_values(
        "Hebb",
        ascending=False
    )
    .head(15)
    .round(4)
    .to_string(index=False)
)

print("\nArquivos gerados:")
print(excel_path.resolve())
print(graph_path.resolve())
print(baseline_metadata_path.resolve())
print(baseline_arrays_path.resolve())

# ============================================================
# 17. ANÁLISE DE SIMILARIDADE ENTRE AS CLASSES
# ============================================================

print("\n" + "=" * 70)
print("SIMILARIDADE DE COSSENO ENTRE AS 13 CLASSES")
print("=" * 70)

print(
    concept_similarity_df
    .round(4)
    .to_string()
)


print("\n" + "=" * 70)
print("DISTÂNCIA DE COSSENO ENTRE AS 13 CLASSES")
print("=" * 70)

print(
    concept_cosine_distance_df
    .round(4)
    .to_string()
)


print("\n" + "=" * 70)
print("CLASSES CONCEITUALMENTE MAIS PRÓXIMAS")
print("=" * 70)

print(
    concept_pairs_df
    .head(15)
    .round(4)
    .to_string(index=False)
)
