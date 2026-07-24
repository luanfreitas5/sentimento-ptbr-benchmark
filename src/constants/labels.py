"""Rótulos de polaridade e mapeamentos legíveis.

A tarefa do benchmark é classificação binária de polaridade a partir da coluna
``polarity`` dos corpora (0 = negativo, 1 = positivo). Reviews neutras foram
descartadas na construção original do dataset.
"""

from __future__ import annotations

from typing import Final

POLARITY_NEGATIVE: Final[int] = 0
POLARITY_POSITIVE: Final[int] = 1

# Mapeamento id -> nome legível (usado em relatórios e figuras).
LABEL_NAMES: Final[dict[int, str]] = {
    POLARITY_NEGATIVE: "negativo",
    POLARITY_POSITIVE: "positivo",
}

# Ordem canônica das classes para matrizes de confusão e métricas.
LABEL_ORDER: Final[tuple[int, ...]] = (POLARITY_NEGATIVE, POLARITY_POSITIVE)
