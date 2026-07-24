"""Consolidação dos resultados do benchmark em um resumo legível.

Lê as matrizes por modelo (``benchmark_<model>.json``) e o teste de McNemar,
produzindo um resumo com desempenho médio in-domain vs. out-of-domain — o
recorte que evidencia a capacidade de generalização entre domínios.
"""

from __future__ import annotations

import logging

import numpy as np

from src.config.paths import ProjectPaths, get_paths
from src.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def _in_out_domain_means(matrix: list[list[float]]) -> tuple[float, float]:
    """Retorna as médias de F1 na diagonal (in-domain) e fora dela (out-domain).

    Parameters
    ----------
    matrix : list[list[float]]
        Matriz ``N x N`` de F1-macro.

    Returns
    -------
    tuple[float, float]
        ``(média_in_domain, média_out_domain)``.

    Examples
    --------
    >>> _in_out_domain_means([[0.9, 0.7], [0.6, 0.8]])
    (0.85, 0.65)
    """
    arr = np.asarray(matrix, dtype=float)
    diagonal = np.diag(arr)
    off_diagonal = arr[~np.eye(arr.shape[0], dtype=bool)]
    in_domain = float(diagonal.mean())
    out_domain = float(off_diagonal.mean()) if off_diagonal.size else float("nan")
    return in_domain, out_domain


def build_benchmark_summary(
    models: list[str],
    paths: ProjectPaths | None = None,
) -> dict[str, dict[str, float]]:
    """Consolida as matrizes dos modelos em um resumo in/out-of-domain.

    Parameters
    ----------
    models : list[str]
        Modelos cujos arquivos ``benchmark_<model>.json`` devem ser lidos.
    paths : ProjectPaths | None, optional
        Caminhos; se ``None``, usa ``get_paths()``.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapa ``modelo -> {"in_domain_f1": ..., "out_domain_f1": ...,
        "generalization_gap": ...}``.

    Raises
    ------
    FileNotFoundError
        Se algum arquivo de benchmark esperado não existir.

    Examples
    --------
    >>> build_benchmark_summary(["tfidf_logreg"])  # doctest: +SKIP
    """
    paths = paths or get_paths()
    summary: dict[str, dict[str, float]] = {}

    for model in models:
        payload = read_json(paths.metrics_dir / f"benchmark_{model}.json")
        in_domain, out_domain = _in_out_domain_means(payload["f1_macro_matrix"])
        summary[model] = {
            "in_domain_f1": round(in_domain, 4),
            "out_domain_f1": round(out_domain, 4),
            "generalization_gap": round(in_domain - out_domain, 4),
        }
        logger.info(
            "[%s] in-domain F1=%.4f | out-domain F1=%.4f | gap=%.4f",
            model,
            in_domain,
            out_domain,
            in_domain - out_domain,
        )

    write_json(summary, paths.metrics_dir / "summary.json")
    return summary
