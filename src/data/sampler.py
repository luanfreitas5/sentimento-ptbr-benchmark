"""Amostragem estratificada dos corpora.

Os corpora somam milhões de linhas; para viabilizar o benchmark (sobretudo o
fine-tuning do BERTimbau) reduzimos cada domínio para um tamanho-alvo,
preservando a proporção de classes de polaridade.
"""

from __future__ import annotations

import logging

import polars as pl

from src.constants.columns import RawColumns

logger = logging.getLogger(__name__)


def stratified_sample(
    df: pl.DataFrame,
    sample_size: int | None,
    *,
    stratify_by: str = RawColumns.POLARITY,
    seed: int = 42,
) -> pl.DataFrame:
    """Amostra ``sample_size`` linhas preservando a proporção de classes.

    Se ``sample_size`` for ``None`` ou maior que o total disponível, o
    DataFrame é retornado inteiro (apenas embaralhado).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de entrada.
    sample_size : int | None
        Número-alvo de linhas na amostra; ``None`` usa todo o DataFrame.
    stratify_by : str, optional
        Coluna categórica para estratificar, by default ``polarity``.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    pl.DataFrame
        Amostra estratificada e embaralhada.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"polarity": [0, 1] * 50, "x": list(range(100))})
    >>> stratified_sample(df, 10).height
    10
    """
    total = df.height
    if sample_size is None or sample_size >= total:
        return df.sample(fraction=1.0, shuffle=True, seed=seed)

    fraction = sample_size / total
    # Amostra cada classe separadamente (estratificação) e reembaralha o todo.
    per_class = [
        group.sample(fraction=fraction, shuffle=True, seed=seed)
        for _, group in df.group_by(stratify_by, maintain_order=True)
    ]
    sampled = pl.concat(per_class).sample(fraction=1.0, shuffle=True, seed=seed)
    logger.info(
        "Amostragem estratificada: %d -> %d linhas (fração=%.4f)",
        total,
        sampled.height,
        fraction,
    )
    return sampled
