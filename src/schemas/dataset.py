"""Contratos de dados para os corpora bruto e processado.

Usa ``pandera`` (backend polars) como contrato explícito e versionado. A
validação ocorre na entrada (após a carga do CSV) e na saída (dataset
processado), pegando desvios de esquema antes que se propaguem.

Examples
--------
>>> import polars as pl
>>> df = pl.DataFrame({"review_text": ["bom"], "polarity": [1], "rating": [5]})
>>> validate_raw(df).height
1
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.polars import Series

from src.constants.datasets import DATASET_NAMES
from src.constants.labels import LABEL_ORDER
from src.exceptions.data import DataValidationError


class RawSchema(pa.DataFrameModel):
    """Contrato mínimo do corpus bruto logo após a leitura do CSV.

    Valida apenas as colunas essenciais (texto, polaridade, nota); colunas
    auxiliares dos CSVs originais são ignoradas.
    """

    review_text: Series[str] = pa.Field(nullable=True)
    polarity: Series[float] = pa.Field(isin=[float(v) for v in LABEL_ORDER], nullable=True)
    rating: Series[float] = pa.Field(ge=1, le=5, nullable=True)

    class Config(BaseConfig):
        """Config do contrato bruto (permite colunas extras dos CSVs)."""

        strict = False
        coerce = True


class ProcessedSchema(pa.DataFrameModel):
    """Contrato do dataset processado pronto para modelagem.

    Após limpeza e amostragem, o texto limpo não pode ser nulo, a polaridade é
    estritamente binária e o domínio pertence ao catálogo conhecido.
    """

    review_text: Series[str] = pa.Field(nullable=False)
    text_clean: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    polarity: Series[int] = pa.Field(isin=list(LABEL_ORDER))
    rating: Series[int] = pa.Field(ge=1, le=5)
    dataset: Series[str] = pa.Field(isin=list(DATASET_NAMES))

    class Config(BaseConfig):
        """Config do contrato processado (rejeita colunas inesperadas)."""

        strict = True
        coerce = True


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Valida o corpus bruto contra ``RawSchema``.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame recém-carregado de um CSV bruto.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado.

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato bruto.
    """
    try:
        return RawSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Corpus bruto inválido:\n{exc}") from exc


def validate_processed(df: pl.DataFrame) -> pl.DataFrame:
    """Valida o dataset processado contra ``ProcessedSchema``.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame processado (limpo e amostrado).

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado.

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato processado.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "review_text": ["Ótimo produto"],
    ...         "text_clean": ["otimo produto"],
    ...         "polarity": [1],
    ...         "rating": [5],
    ...         "dataset": ["b2w"],
    ...     }
    ... )
    >>> validate_processed(df).height
    1
    """
    try:
        return ProcessedSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Dataset processado inválido:\n{exc}") from exc
