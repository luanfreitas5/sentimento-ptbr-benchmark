"""Métricas de classificação de polaridade com incerteza por bootstrap.

O headline do benchmark é o **F1-macro**: por darem forte peso à classe
positiva, os corpora exigem uma métrica que trate ambas as classes
igualmente. Nunca reportamos um ponto isolado — sempre acompanhado de um
intervalo de confiança por reamostragem.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants.labels import LABEL_ORDER


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Calcula as métricas de classificação binária de polaridade.

    Parameters
    ----------
    y_true : Sequence[int] | np.ndarray
        Rótulos verdadeiros (0/1).
    y_pred : Sequence[int] | np.ndarray
        Rótulos previstos (0/1).

    Returns
    -------
    dict[str, float]
        Chaves: ``accuracy``, ``f1_macro``, ``f1_positive``, ``precision_macro``,
        ``recall_macro``.

    Examples
    --------
    >>> m = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    >>> round(m["accuracy"], 2)
    0.75
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # zero_division aceita 0/1 em runtime; o stub inferido do sklearn (sem
    # anotação própria) só permite "warn", daí a supressão pontual abaixo.
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        "f1_positive": float(
            f1_score(y_true_arr, y_pred_arr, pos_label=LABEL_ORDER[1], zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
    }


def bootstrap_confidence_interval(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    metric: str = "f1_macro",
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estima o IC de uma métrica por bootstrap sobre o conjunto de teste.

    Parameters
    ----------
    y_true : Sequence[int] | np.ndarray
        Rótulos verdadeiros.
    y_pred : Sequence[int] | np.ndarray
        Rótulos previstos.
    metric : str, optional
        Métrica a reamostrar (chave de ``compute_classification_metrics``),
        by default ``f1_macro``.
    n_samples : int, optional
        Número de reamostragens, by default 1000.
    confidence : float, optional
        Nível de confiança, by default 0.95.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    tuple[float, float, float]
        ``(ponto, limite_inferior, limite_superior)`` da métrica.

    Examples
    --------
    >>> point, lo, hi = bootstrap_confidence_interval([0, 1] * 20, [0, 1] * 20, n_samples=100)
    >>> lo <= point <= hi
    True
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true_arr)

    point = compute_classification_metrics(y_true_arr, y_pred_arr)[metric]

    scores = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        scores[i] = compute_classification_metrics(y_true_arr[idx], y_pred_arr[idx])[metric]

    alpha = (1 - confidence) / 2
    lower = float(np.quantile(scores, alpha))
    upper = float(np.quantile(scores, 1 - alpha))
    return point, lower, upper
