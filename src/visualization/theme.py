"""Tema e utilitários de figura compartilhados do projeto.

Centraliza a paleta e o estilo para que todas as figuras tenham aparência
consistente, e padroniza o salvamento em ``.png`` (300 dpi) + ``.svg``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Paleta consistente do projeto (verde = positivo, vermelho = negativo).
PALETTE: Final[dict[str, str]] = {
    "positivo": "#2a9d8f",
    "negativo": "#e76f51",
    "neutro": "#264653",
    "accent": "#e9c46a",
}
SEQUENTIAL_CMAP: Final[str] = "viridis"
FIGURE_DPI: Final[int] = 300


def apply_theme() -> None:
    """Aplica o tema global (seaborn ``whitegrid`` + parâmetros de fonte).

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"


def save_figure(fig: Figure, path: Path) -> None:
    """Salva a figura em ``.png`` (300 dpi) e ``.svg`` no mesmo caminho base.

    Parameters
    ----------
    fig : Figure
        Figura do matplotlib a salvar.
    path : Path
        Caminho base (a extensão é substituída por ``.png`` e ``.svg``).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pathlib import Path
    >>> fig, _ = plt.subplots()
    >>> save_figure(fig, Path("reports/figures/exemplo.png"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
