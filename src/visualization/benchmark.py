"""Heatmap da matriz cross-dataset (treino x teste) — o diferencial do projeto.

Cada célula ``(i, j)`` é o F1-macro de um modelo treinado no domínio ``i`` e
avaliado no domínio ``j``. A diagonal mede o desempenho in-domain; as células
fora da diagonal medem a generalização out-of-domain (transferência).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.constants.datasets import DATASET_CATALOG
from src.visualization.theme import SEQUENTIAL_CMAP


def plot_cross_dataset_matrix(
    matrix: Sequence[Sequence[float]],
    domains: Sequence[str],
    *,
    model_name: str,
    metric_name: str = "F1-macro",
) -> Figure:
    """Plota a matriz cross-dataset de um modelo como heatmap.

    Parameters
    ----------
    matrix : Sequence[Sequence[float]]
        Matriz ``N x N`` de valores da métrica (linha = treino, coluna = teste).
    domains : Sequence[str]
        Identificadores dos domínios, na ordem das linhas/colunas.
    model_name : str
        Nome do modelo (usado no título).
    metric_name : str, optional
        Nome da métrica exibida, by default ``"F1-macro"``.

    Returns
    -------
    Figure
        Figura do matplotlib pronta para salvar.

    Examples
    --------
    >>> fig = plot_cross_dataset_matrix(
    ...     [[0.9, 0.7], [0.6, 0.88]], ["b2w", "olist"], model_name="tfidf_logreg"
    ... )
    >>> fig is not None
    True
    """
    data = np.asarray(matrix, dtype=float)
    labels = [DATASET_CATALOG[d].display_name if d in DATASET_CATALOG else d for d in domains]

    fig, ax = plt.subplots(figsize=(1.4 * len(domains) + 2, 1.2 * len(domains) + 2))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap=SEQUENTIAL_CMAP,
        vmin=0.5,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": metric_name},
        ax=ax,
    )
    ax.set_title(f"Benchmark cross-dataset — {model_name} ({metric_name})")
    ax.set_xlabel("Domínio de teste")
    ax.set_ylabel("Domínio de treino")
    return fig
