"""Formulário desktop para as operações mais comuns do projeto CAPEX.

A camada :class:`MenuService` não depende do Tk, o que mantém as operações
reutilizáveis e permite testar o formulário em ambientes sem interface gráfica.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classifier import ClassifierModel, ProjectClassification, classify_project
from .config import CONCEPT_MATRIX_VERSION, ETA, OUTPUT_DIR
from .data import load_concept_matrices, load_project_partitions
from .hebbian import train_hebbian
from .schema import CLASSES, FEATURES
from .visualization import save_pca_plot


DEFAULT_THRESHOLDS = {"minimum_similarity": 0.70, "minimum_margin": 0.10}


@dataclass(frozen=True)
class TrainingResult:
    """Resumo do artefato produzido pelo treino do protótipo de Segurança."""

    path: Path
    project_count: int
    history_length: int


class MenuService:
    """Executa as ações do menu sem acoplá-las aos componentes visuais."""

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def classify(self, project_code, features) -> ProjectClassification:
        """Classifica um vetor usando o baseline conceitual oficial."""
        _, _, concepts = load_concept_matrices()
        model = ClassifierModel(concepts, tuple(CLASSES), CONCEPT_MATRIX_VERSION)
        return classify_project(
            project_code=project_code,
            features=features,
            model=model,
            thresholds=DEFAULT_THRESHOLDS,
        )

    def train(self) -> TrainingResult:
        """Treina e persiste o protótipo histórico da classe Segurança.

        O corpus disponível não contém exemplos das outras doze classes. Por
        isso esta ação não afirma realizar treino multiclasse: ela atualiza
        somente o protótipo de Segurança, preservando essa limitação no nome e
        nos metadados do artefato.
        """
        _, _, concepts = load_concept_matrices()
        training = load_project_partitions()["train"]
        security = training.loc[
            training["real_class"] == CLASSES[0], FEATURES
        ].to_numpy(dtype=float)
        if not len(security):
            raise ValueError("não há projetos de Segurança disponíveis para treino")
        weights, history = train_hebbian(security, concepts[0], eta=ETA)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "prototipo_hebbiano_seguranca.npz"
        np.savez(
            path,
            weights=weights,
            history=history,
            class_name=CLASSES[0],
            model_version=CONCEPT_MATRIX_VERSION,
            eta=ETA,
            project_count=len(security),
        )
        return TrainingResult(path, len(security), len(history))

    def create_3d_chart(self) -> Path:
        """Gera o gráfico PCA 3D com os projetos de treino e validação."""
        _, _, concepts = load_concept_matrices()
        partitions = load_project_partitions()
        projects = np.vstack(
            [
                table.loc[:, FEATURES].to_numpy(dtype=float)
                for table in partitions.values()
                if len(table)
            ]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "pca_3d_menu.png"
        save_pca_plot(
            path,
            concepts,
            projects,
            CLASSES,
            data_version="partições versionadas",
            model_version=CONCEPT_MATRIX_VERSION,
        )
        return path


class MenuApp:
    """Janela principal e formulário rolável das 42 características."""

    def __init__(self, root, service=None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.service = service or MenuService()
        root.title("Classificador de projetos CAPEX")
        root.geometry("760x520")
        root.minsize(640, 420)

        container = ttk.Frame(root, padding=24)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="Classificação de projetos CAPEX",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(pady=(20, 8))
        ttk.Label(
            container,
            text="Escolha uma operação para continuar.",
        ).pack(pady=(0, 28))

        actions = ttk.Frame(container)
        actions.pack(fill="x", padx=80)
        for label, command in (
            ("Classificar um projeto", self.open_classification_form),
            ("Treinar a rede", self.train_network),
            ("Visualizar gráfico 3D", self.show_3d_chart),
        ):
            ttk.Button(actions, text=label, command=command).pack(
                fill="x", pady=8, ipady=10
            )
        ttk.Separator(container).pack(fill="x", pady=28)
        ttk.Label(
            container,
            text=(
                "Nota: o treino disponível atualiza apenas o protótipo de Segurança; "
                "o gráfico 3D é exploratório."
            ),
            wraplength=620,
            justify="center",
        ).pack()

    def open_classification_form(self):
        import tkinter as tk
        from tkinter import ttk

        window = tk.Toplevel(self.root)
        window.title("Classificar um projeto")
        window.geometry("820x650")
        header = ttk.Frame(window, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Código do projeto:").pack(side="left")
        code = ttk.Entry(header, width=30)
        code.pack(side="left", padx=8)

        canvas = tk.Canvas(window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        fields = ttk.Frame(canvas, padding=12)
        fields.bind(
            "<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=fields, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        values = {}
        for row, feature in enumerate(FEATURES):
            ttk.Label(fields, text=feature).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            variable = tk.DoubleVar(value=0.0)
            ttk.Spinbox(
                fields, from_=0.0, to=1.0, increment=0.05,
                textvariable=variable, width=8,
            ).grid(row=row, column=1, padx=12, pady=3)
            values[feature] = variable
        ttk.Button(
            fields,
            text="Classificar",
            command=lambda: self._classify(code.get(), values),
        ).grid(row=len(FEATURES), column=0, columnspan=2, sticky="ew", pady=16)

    def _classify(self, code, variables):
        from tkinter import messagebox

        try:
            result = self.service.classify(
                code, {name: variable.get() for name, variable in variables.items()}
            )
        except (TypeError, ValueError) as error:
            messagebox.showerror("Dados inválidos", str(error), parent=self.root)
            return
        messagebox.showinfo(
            "Resultado da classificação",
            f"Classe: {result.predicted_class}\n"
            f"Similaridade: {result.similarity:.4f}\n"
            f"Segunda classe: {result.second_class}\n"
            f"Margem: {result.margin:.4f}\n"
            f"Status: {result.status}",
            parent=self.root,
        )

    def train_network(self):
        from tkinter import messagebox

        try:
            result = self.service.train()
        except (OSError, ValueError) as error:
            messagebox.showerror("Falha no treino", str(error), parent=self.root)
            return
        messagebox.showinfo(
            "Treino concluído",
            f"Protótipo treinado com {result.project_count} projetos.\n"
            f"Artefato salvo em:\n{result.path.resolve()}",
            parent=self.root,
        )

    def show_3d_chart(self):
        from tkinter import messagebox

        try:
            path = self.service.create_3d_chart()
            self.root.tk.call("tk", "scaling")  # confirma que a janela segue ativa
            import webbrowser

            webbrowser.open(path.resolve().as_uri())
        except (OSError, ValueError) as error:
            messagebox.showerror("Falha ao gerar gráfico", str(error), parent=self.root)


def main():
    """Cria a janela somente quando o lançador é executado diretamente."""
    import tkinter as tk

    root = tk.Tk()
    MenuApp(root)
    root.mainloop()
