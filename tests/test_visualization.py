"""Testes das figuras do benchmark (checagem de fumaça, sem I/O de arquivo)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend headless para testes

import pytest
from matplotlib.figure import Figure

from src.visualization.benchmark import plot_cross_dataset_matrix
from src.visualization.confusion_matrix import plot_confusion_matrix


@pytest.mark.smoke
def test_cross_dataset_matrix_returns_figure() -> None:
    """A matriz cross-dataset produz uma figura do matplotlib."""
    fig = plot_cross_dataset_matrix(
        [[0.9, 0.7], [0.6, 0.88]],
        ["b2w", "olist"],
        model_name="tfidf_logreg",
    )
    assert isinstance(fig, Figure)


def test_confusion_matrix_returns_figure() -> None:
    """A matriz de confusão produz uma figura do matplotlib."""
    fig = plot_confusion_matrix([[40, 10], [5, 45]])
    assert isinstance(fig, Figure)
