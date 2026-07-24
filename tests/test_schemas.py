"""Testes dos contratos de dados (pandera)."""

from __future__ import annotations

import polars as pl
import pytest

from src.exceptions.data import DataValidationError
from src.schemas.dataset import validate_processed, validate_raw


def test_validate_raw_accepts_valid_frame(raw_domain_df: pl.DataFrame) -> None:
    """Um corpus bruto válido passa pelo contrato."""
    assert validate_raw(raw_domain_df).height == raw_domain_df.height


def test_validate_raw_rejects_out_of_range_polarity() -> None:
    """Polaridade fora de {0, 1} viola o contrato bruto."""
    polarity_df = pl.DataFrame({"review_text": ["x"], "polarity": [2.0], "rating": [5.0]})
    with pytest.raises(DataValidationError):
        validate_raw(polarity_df)


def _valid_processed() -> pl.DataFrame:
    """DataFrame processado mínimo e válido."""
    return pl.DataFrame(
        {
            "review_text": ["Ótimo produto"],
            "text_clean": ["otimo produto"],
            "polarity": [1],
            "rating": [5],
            "dataset": ["b2w"],
        }
    )


def test_validate_processed_accepts_valid_frame() -> None:
    """O dataset processado válido passa pelo contrato."""
    assert validate_processed(_valid_processed()).height == 1


def test_validate_processed_rejects_empty_text() -> None:
    """Texto limpo vazio viola o contrato processado."""
    processed_df = _valid_processed().with_columns(pl.lit("").alias("text_clean"))
    with pytest.raises(DataValidationError):
        validate_processed(processed_df)


def test_validate_processed_rejects_unknown_domain() -> None:
    """Um domínio fora do catálogo viola o contrato processado."""
    processed_df = _valid_processed().with_columns(pl.lit("desconhecido").alias("dataset"))
    with pytest.raises(DataValidationError):
        validate_processed(processed_df)
