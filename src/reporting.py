"""Exportação explícita de tabelas; importar este módulo não grava arquivos."""
from pathlib import Path
import pandas as pd


def export_excel(path, sheets):
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31])
    return path
