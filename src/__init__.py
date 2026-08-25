"""Componentes do classificador de projetos CAPEX.

As exportações públicas são carregadas sob demanda para que os módulos legados
que definem o esquema possam continuar importando submódulos de ``src``.
"""

__all__ = [
    "ClassifierModel",
    "ProjectClassification",
    "classify_project",
    "register_human_review",
]


def __getattr__(name):
    if name in __all__:
        from src import classifier

        return getattr(classifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
