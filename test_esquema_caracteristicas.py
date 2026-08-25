"""Contratos mínimos do dicionário operacional de características."""
import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "docs" / "esquema_caracteristicas_X01_X42.json"
REQUIRED_FIELDS = {
    "codigo",
    "nome",
    "descricao_operacional",
    "evidencia_necessaria",
    "escala_permitida",
    "significado_niveis",
    "regra_informacao_desconhecida",
    "exemplos",
    "responsavel_validacao",
}


def load_dictionary():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_dictionary_defines_all_features_and_required_fields():
    document = load_dictionary()
    features = document["caracteristicas"]

    assert [feature["codigo"] for feature in features] == [f"X{i:02d}" for i in range(1, 43)]
    for feature in features:
        assert REQUIRED_FIELDS <= feature.keys()
        assert feature["escala_permitida"] == [0, 0.5, 1, "UNKNOWN"]
        assert set(feature["significado_niveis"]) == {"0", "0.5", "1", "UNKNOWN"}
        assert set(feature["exemplos"]) == {"positivo", "negativo"}
        assert all(feature[field] for field in REQUIRED_FIELDS - {"escala_permitida"})


def test_unknown_is_distinct_from_confirmed_absence():
    document = load_dictionary()
    policy = document["politica_ausencia_desconhecido"]

    assert "evidência" in policy["ausencia"]
    assert "insuficiente" in policy["desconhecido"]
    assert "não pode ser imputado automaticamente como 0" in policy["restricao"]
    for feature in document["caracteristicas"]:
        assert "Nunca converter UNKNOWN em 0" in feature["regra_informacao_desconhecida"]


def test_overlap_reviews_and_expert_review_gates_are_recorded():
    document = load_dictionary()
    reviewed_groups = {
        frozenset(review["caracteristicas"])
        for review in document["revisao_sobreposicoes"]
    }
    assert {
        frozenset(("X01", "X02", "X04", "X38")),
        frozenset(("X03", "X33")),
        frozenset(("X34", "X38")),
        frozenset(("X39", "X40", "X41")),
        frozenset(("X11", "X12", "X13", "X28")),
    } <= reviewed_groups

    reviews = {review["artefato"]: review for review in document["revisao_especialistas"]}
    assert set(reviews) == {"C32", "C_EXTRA"}
    assert all(review["status"] == "SUBMETIDA_PENDENTE" for review in reviews.values())
    assert document["controle_versoes"]["alteracoes"]
