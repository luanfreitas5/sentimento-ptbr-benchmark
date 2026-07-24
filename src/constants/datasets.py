"""Identificadores e metadados dos cinco domínios do benchmark.

Cada domínio é um corpus rotulado de sentimento em PT-BR. A avaliação
cross-dataset treina em um domínio e testa nos demais, revelando o quanto
cada modelo generaliza fora do domínio de origem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class DatasetInfo:
    """Metadados de um domínio (corpus) do benchmark.

    Attributes
    ----------
    name : str
        Identificador curto usado em caminhos e relatórios (ex.: ``b2w``).
    display_name : str
        Nome legível para figuras e documentação.
    domain : str
        Domínio de conteúdo (ex.: e-commerce, apps, filmes).
    """

    name: str
    display_name: str
    domain: str


DATASET_CATALOG: Final[dict[str, DatasetInfo]] = {
    "b2w": DatasetInfo("b2w", "B2W", "e-commerce (varejo)"),
    "buscape": DatasetInfo("buscape", "Buscapé", "comparador de preços"),
    "olist": DatasetInfo("olist", "Olist", "marketplace"),
    "utlc_apps": DatasetInfo("utlc_apps", "UTLC-Apps", "avaliações de apps"),
    "utlc_movies": DatasetInfo("utlc_movies", "UTLC-Movies", "avaliações de filmes"),
}

# Ordem canônica dos domínios (usada nos eixos da matriz cross-dataset).
DATASET_NAMES: Final[tuple[str, ...]] = tuple(DATASET_CATALOG.keys())
