"""Avaliação de um modelo em um conjunto de teste.

Produz o pacote de evidências exigido pelo *senior bar*: métrica headline com
intervalo de confiança, demais métricas e matriz de confusão — nunca um número
solto.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix

from src.constants.labels import LABEL_ORDER
from src.metrics.classification import (
    bootstrap_confidence_interval,
    compute_classification_metrics,
)
from src.models.base import SentimentClassifier

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Resultado da avaliação de um modelo em um conjunto de teste.

    Attributes
    ----------
    metrics : dict[str, float]
        Métricas pontuais (accuracy, f1_macro, ...).
    primary_metric : str
        Nome da métrica headline (ex.: ``f1_macro``).
    ci_lower, ci_upper : float
        Limites do intervalo de confiança da métrica headline.
    confusion : list[list[int]]
        Matriz de confusão na ordem de ``LABEL_ORDER``.
    y_pred : np.ndarray
        Rótulos previstos (guardados para o teste de McNemar).
    """

    metrics: dict[str, float]
    primary_metric: str
    ci_lower: float
    ci_upper: float
    confusion: list[list[int]]
    y_pred: np.ndarray = field(repr=False)

    @property
    def headline(self) -> float:
        """Valor pontual da métrica headline."""
        return self.metrics[self.primary_metric]

    def summary(self) -> str:
        """Retorna um resumo legível ``métrica: valor [IC 95%]``."""
        return (
            f"{self.primary_metric}={self.headline:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


def evaluate_model(
    model: SentimentClassifier,
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    primary_metric: str = "f1_macro",
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> EvaluationResult:
    """Avalia um modelo treinado em um conjunto de teste.

    Parameters
    ----------
    model : SentimentClassifier
        Modelo já treinado.
    texts : Sequence[str]
        Textos de teste.
    labels : Sequence[int]
        Rótulos verdadeiros (0/1).
    primary_metric : str, optional
        Métrica headline, by default ``f1_macro``.
    bootstrap_samples : int, optional
        Reamostragens para o IC, by default 1000.
    confidence : float, optional
        Nível de confiança do IC, by default 0.95.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    EvaluationResult
        Métricas, IC da headline, matriz de confusão e predições.

    Examples
    --------
    >>> from src.config.settings import TfidfLogRegConfig
    >>> from src.models.tfidf_logreg import TfidfLogRegClassifier
    >>> clf = TfidfLogRegClassifier(TfidfLogRegConfig()).fit(
    ...     ["ótimo", "ruim", "adorei", "detestei"], [1, 0, 1, 0]
    ... )
    >>> res = evaluate_model(clf, ["bom", "péssimo"], [1, 0], bootstrap_samples=50)
    >>> 0.0 <= res.headline <= 1.0
    True
    """
    y_pred = model.predict(texts)
    metrics = compute_classification_metrics(labels, y_pred)
    _, ci_lower, ci_upper = bootstrap_confidence_interval(
        labels,
        y_pred,
        metric=primary_metric,
        n_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    matrix = confusion_matrix(labels, y_pred, labels=list(LABEL_ORDER)).tolist()

    result = EvaluationResult(
        metrics=metrics,
        primary_metric=primary_metric,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confusion=matrix,
        y_pred=np.asarray(y_pred),
    )
    logger.info("Avaliação concluída: %s", result.summary())
    return result
