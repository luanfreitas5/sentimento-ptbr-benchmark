"""Cálculo de métricas de classificação com quantificação de incerteza.

Módulos
-------
classification
    Métricas de polaridade (F1-macro, acurácia, precisão/recall) e intervalos
    de confiança por bootstrap.
"""

from src.metrics.classification import (
    bootstrap_confidence_interval,
    compute_classification_metrics,
)

__all__ = ["bootstrap_confidence_interval", "compute_classification_metrics"]
