import ast
from pathlib import Path


def test_main_possui_guarda_de_execucao():
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
    assert 'if __name__ == "__main__":' in Path("main.py").read_text(encoding="utf-8")


def test_modulos_esperados_existem():
    expected = {"config.py", "schema.py", "data.py", "preprocessing.py", "classifier.py", "hebbian.py", "evaluation.py", "visualization.py", "reporting.py"}
    assert expected <= {path.name for path in Path("src").glob("*.py")}
