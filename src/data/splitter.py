"""Divisão treino/teste estratificada e determinística por domínio.

Cada domínio é dividido em treino e teste. O conjunto de teste serve para a
avaliação in-domain (na diagonal da matriz cross-dataset) e como alvo das
avaliações out-of-domain (fora da diagonal).
"""

from __future__ import annotations

import logging

import polars as pl
from sklearn.model_selection import train_test_split

from src.constants.columns import RawColumns

logger = logging.getLogger(__name__)


def train_test_split_domain(
    df: pl.DataFrame,
    *,
    test_size: float = 0.2,
    stratify: bool = True,
    target: str = RawColumns.POLARITY,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide um domínio em treino e teste de forma estratificada.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame processado de um único domínio.
    test_size : float, optional
        Proporção destinada ao teste, by default 0.2.
    stratify : bool, optional
        Se ``True``, estratifica pela coluna-alvo, by default True.
    target : str, optional
        Coluna-alvo para estratificação, by default ``polarity``.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Par ``(treino, teste)``.

    Raises
    ------
    ValueError
        Se o DataFrame tiver menos de 2 linhas para dividir.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"polarity": [0, 1] * 25, "text_clean": ["x"] * 50})
    >>> tr, te = train_test_split_domain(df, test_size=0.2)
    >>> tr.height, te.height
    (40, 10)
    """
    if df.height < 2:
        raise ValueError(f"DataFrame com {df.height} linha(s) é insuficiente para dividir.")

    indices = list(range(df.height))
    stratify_labels = df[target].to_list() if stratify else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=seed,
        shuffle=True,
    )

    train_df = df[train_idx]
    test_df = df[test_idx]
    logger.info(
        "Split: treino=%d, teste=%d (test_size=%.2f, estratificado=%s)",
        train_df.height,
        test_df.height,
        test_size,
        stratify,
    )
    return train_df, test_df
