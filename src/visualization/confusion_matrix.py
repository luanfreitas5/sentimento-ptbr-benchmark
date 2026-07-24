"""Heatmap de matriz de confusão."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.constants.labels import LABEL_NAMES, LABEL_ORDER


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    *,
    title: str = "Matriz de Confusão",
    normalize: bool = True,
) -> Figure:
    """Plota uma matriz de confusão como heatmap anotado.

    Parameters
    ----------
    matrix : Sequence[Sequence[int]]
        Matriz de confusão 2x2 na ordem de ``LABEL_ORDER``.
    title : str, optional
        Título da figura, by default ``"Matriz de Confusão"``.
    normalize : bool, optional
        Se ``True``, normaliza por linha (recall por classe), by default True.

    Returns
    -------
    Figure
        Figura do matplotlib pronta para salvar.

    Examples
    --------
    >>> fig = plot_confusion_matrix([[40, 10], [5, 45]])
    >>> fig is not None
    True
    """
    data = np.asarray(matrix, dtype=float)
    fmt = ".2f"
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    else:
        fmt = ".0f"

    tick_labels = [LABEL_NAMES[label] for label in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
    )
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    return fig
