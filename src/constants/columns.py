"""Nomes de colunas dos corpora brutos e do dataset processado.

Os cinco CSVs de origem compartilham o mesmo esquema (mais uma coluna
``dataset`` no arquivo concatenado). Centralizar os nomes evita *strings
mágicas* espalhadas pelo código.
"""

from __future__ import annotations

from typing import Final


class RawColumns:
    """Colunas presentes nos CSVs brutos dos corpora de sentimento."""

    ORIGINAL_INDEX: Final = "original_index"
    REVIEW_TEXT: Final = "review_text"
    REVIEW_TEXT_PROCESSED: Final = "review_text_processed"
    REVIEW_TEXT_TOKENIZED: Final = "review_text_tokenized"
    POLARITY: Final = "polarity"
    RATING: Final = "rating"
    KFOLD_POLARITY: Final = "kfold_polarity"
    KFOLD_RATING: Final = "kfold_rating"
    DATASET: Final = "dataset"  # presente apenas no arquivo concatenated.csv


# Coluna de texto limpo produzida por src/preprocessing/text.py.
TEXT_CLEAN: Final = "text_clean"

# Colunas mínimas exigidas na carga (as demais são opcionais/descartadas).
RAW_COLUMNS: tuple[str, ...] = (
    RawColumns.REVIEW_TEXT,
    RawColumns.POLARITY,
    RawColumns.RATING,
)

# Colunas do dataset processado pronto para modelagem.
PROCESSED_COLUMNS: tuple[str, ...] = (
    RawColumns.REVIEW_TEXT,
    TEXT_CLEAN,
    RawColumns.POLARITY,
    RawColumns.RATING,
    RawColumns.DATASET,
)
