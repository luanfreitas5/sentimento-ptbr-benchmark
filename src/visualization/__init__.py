"""Figuras do benchmark (matriz cross-dataset, matrizes de confusão).

Todas as figuras são salvas em ``reports/figures`` em ``.png`` (300 dpi) e
``.svg``, com eixos rotulados e títulos em pt-BR.

Módulos
-------
theme
    Paleta e estilo compartilhados do projeto.
confusion_matrix
    Heatmap de matriz de confusão.
benchmark
    Heatmap da matriz cross-dataset (treino x teste) de F1-macro.
"""

from src.visualization.benchmark import plot_cross_dataset_matrix
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.theme import apply_theme, save_figure

__all__ = [
    "apply_theme",
    "plot_confusion_matrix",
    "plot_cross_dataset_matrix",
    "save_figure",
]
