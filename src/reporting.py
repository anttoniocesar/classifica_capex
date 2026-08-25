"""Exportação explícita de tabelas; importar este módulo não grava arquivos."""
from dataclasses import asdict
from pathlib import Path
import pandas as pd


def decisions_to_table(decisions):
    """Converte decisões, incluindo os limiares usados, para exportação."""
    return pd.DataFrame(asdict(decision) for decision in decisions)


def export_excel(path, sheets):
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31])
    return path
