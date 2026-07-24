"""Carregamento e preparação dos corpora brutos de sentimento.

Lê os CSVs com polars (lazy), seleciona as colunas essenciais, valida o
contrato bruto, remove nulos/duplicatas, aplica a limpeza de texto e devolve um
DataFrame já anotado com o domínio de origem.
"""

from __future__ import annotations

import logging

import polars as pl

from src.config.paths import ProjectPaths, get_paths
from src.constants.columns import TEXT_CLEAN, RawColumns
from src.exceptions.data import DatasetNotFoundError
from src.preprocessing.text import clean_text
from src.schemas.dataset import validate_raw

logger = logging.getLogger(__name__)


def load_raw_domain(domain: str, paths: ProjectPaths | None = None) -> pl.DataFrame:
    """Carrega o CSV bruto de um domínio e valida o contrato mínimo.

    Parameters
    ----------
    domain : str
        Identificador do domínio (ex.: ``b2w``, ``olist``).
    paths : ProjectPaths | None, optional
        Caminhos do projeto; se ``None``, usa ``get_paths()``.

    Returns
    -------
    pl.DataFrame
        Colunas ``review_text``, ``polarity`` e ``rating`` validadas.

    Raises
    ------
    DatasetNotFoundError
        Se o CSV do domínio não existir em ``data/raw``.

    Examples
    --------
    >>> df = load_raw_domain("b2w")  # doctest: +SKIP
    """
    paths = paths or get_paths()
    csv_path = paths.raw_dataset(domain)
    if not csv_path.exists():
        raise DatasetNotFoundError(
            f"Corpus '{domain}' não encontrado em {csv_path}. "
            "Baixe o dataset do Kaggle para data/raw/."
        )

    logger.info("Carregando corpus bruto '%s' de %s", domain, csv_path.name)
    lazy = pl.scan_csv(csv_path, infer_schema_length=10_000).select(
        pl.col(RawColumns.REVIEW_TEXT).cast(pl.Utf8, strict=False),
        pl.col(RawColumns.POLARITY).cast(pl.Float64, strict=False),
        pl.col(RawColumns.RATING).cast(pl.Float64, strict=False),
    )
    return validate_raw(lazy.collect())


def prepare_domain(
    domain: str,
    *,
    min_chars: int = 3,
    paths: ProjectPaths | None = None,
) -> pl.DataFrame:
    """Prepara um domínio: valida, limpa, remove nulos/curtos e anota origem.

    Parameters
    ----------
    domain : str
        Identificador do domínio.
    min_chars : int, optional
        Comprimento mínimo do texto limpo para manter a linha, by default 3.
    paths : ProjectPaths | None, optional
        Caminhos do projeto; se ``None``, usa ``get_paths()``.

    Returns
    -------
    pl.DataFrame
        Colunas ``review_text``, ``text_clean``, ``polarity`` (int), ``rating``
        (int) e ``dataset``, sem nulos nem duplicatas de texto limpo.

    Examples
    --------
    >>> df = prepare_domain("olist")  # doctest: +SKIP
    """

    prepared = (
        load_raw_domain(domain, paths=paths)
        .drop_nulls([RawColumns.REVIEW_TEXT, RawColumns.POLARITY, RawColumns.RATING])
        .with_columns(
            pl.col(RawColumns.REVIEW_TEXT)
            .map_elements(clean_text, return_dtype=pl.Utf8)
            .alias(TEXT_CLEAN),
            pl.col(RawColumns.POLARITY).cast(pl.Int64),
            pl.col(RawColumns.RATING).cast(pl.Int64),
            pl.lit(domain).alias(RawColumns.DATASET),
        )
        .filter(pl.col(TEXT_CLEAN).str.len_chars() >= min_chars)
        .unique(subset=[TEXT_CLEAN], keep="first")
    )

    positive_pct = (
        100 * float(prepared[RawColumns.POLARITY].sum()) / prepared.height
        if prepared.height
        else 0.0
    )
    logger.info(
        "Domínio '%s' preparado: %d linhas (positivos=%.1f%%)",
        domain,
        prepared.height,
        positive_pct,
    )
    return prepared
