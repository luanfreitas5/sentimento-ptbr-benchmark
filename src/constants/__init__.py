"""Constantes, enums e valores padrão do projeto.

Módulos
-------
columns
    Nomes das colunas dos corpora brutos e processados.
labels
    Rótulos de polaridade e seus mapeamentos legíveis.
datasets
    Identificadores e metadados dos cinco domínios do benchmark.
"""

from src.constants.columns import PROCESSED_COLUMNS, RAW_COLUMNS, RawColumns
from src.constants.datasets import DATASET_CATALOG, DATASET_NAMES, DatasetInfo
from src.constants.labels import LABEL_NAMES, POLARITY_NEGATIVE, POLARITY_POSITIVE

__all__ = [
    "DATASET_CATALOG",
    "DATASET_NAMES",
    "LABEL_NAMES",
    "POLARITY_NEGATIVE",
    "POLARITY_POSITIVE",
    "PROCESSED_COLUMNS",
    "RAW_COLUMNS",
    "DatasetInfo",
    "RawColumns",
]
