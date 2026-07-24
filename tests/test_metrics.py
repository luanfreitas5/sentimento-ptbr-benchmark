"""Testes das métricas de classificação e do IC por bootstrap."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.metrics.classification import (
    bootstrap_confidence_interval,
    compute_classification_metrics,
)


@pytest.mark.smoke
def test_perfect_predictions_score_one() -> None:
    """Predições perfeitas dão métricas iguais a 1."""
    metrics = compute_classification_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_metrics_keys_present() -> None:
    """O dicionário de métricas contém todas as chaves esperadas."""
    metrics = compute_classification_metrics([0, 1], [1, 1])
    assert {
        "accuracy",
        "f1_macro",
        "f1_positive",
        "precision_macro",
        "recall_macro",
    } <= set(metrics)


def test_bootstrap_interval_contains_point_estimate() -> None:
    """O ponto estimado fica dentro do intervalo de confiança."""
    y_true = [0, 1] * 30
    y_pred = [0, 1] * 30
    point, lower, upper = bootstrap_confidence_interval(y_true, y_pred, n_samples=200)
    assert lower <= point <= upper


@given(
    labels=st.lists(st.integers(min_value=0, max_value=1), min_size=4, max_size=50),
)
def test_metrics_bounded_between_zero_and_one(labels: list[int]) -> None:
    """Invariante: toda métrica fica no intervalo [0, 1]."""
    preds = [1 - label for label in labels]  # sempre erra: força variação
    metrics = compute_classification_metrics(labels, preds)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
