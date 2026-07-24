"""Testes do baseline TF-IDF + LogReg (unit, persistência, comportamentais)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.exceptions.model import ModelNotFittedError
from src.models.persistence import load_model
from src.models.tfidf_logreg import TfidfLogRegClassifier
from tests.conftest import make_test_config


@pytest.mark.smoke
def test_predict_returns_valid_labels(
    trained_baseline: TfidfLogRegClassifier,
) -> None:
    """As predições estão no espaço de rótulos {0, 1}."""
    preds = trained_baseline.predict(["produto muito bom", "produto horrível"])
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_shape_and_normalization(
    trained_baseline: TfidfLogRegClassifier,
) -> None:
    """As probabilidades têm shape (n, 2) e somam 1 por linha."""
    proba = trained_baseline.predict_proba(["ótimo", "péssimo"])
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_before_fit_raises() -> None:
    """Prever antes de treinar levanta ``ModelNotFittedError``."""
    model = TfidfLogRegClassifier(make_test_config())
    with pytest.raises(ModelNotFittedError):
        model.predict(["x"])


def test_save_and_load_roundtrip(
    trained_baseline: TfidfLogRegClassifier,
    tmp_path: Path,
) -> None:
    """Um modelo salvo e recarregado produz as mesmas predições."""
    texts = ["produto excelente", "produto ruim"]
    before = trained_baseline.predict(texts)

    trained_baseline.save(tmp_path)
    reloaded = TfidfLogRegClassifier.load(tmp_path)
    after = reloaded.predict(texts)

    np.testing.assert_array_equal(before, after)


def test_persistence_load_model_dispatch(
    trained_baseline: TfidfLogRegClassifier,
    tmp_path: Path,
) -> None:
    """O dispatcher ``load_model`` recarrega o baseline pelo nome."""
    trained_baseline.save(tmp_path)
    reloaded = load_model("tfidf_logreg", tmp_path)
    assert isinstance(reloaded, TfidfLogRegClassifier)
    assert reloaded.predict(["ótimo produto"]).shape == (1,)


@pytest.mark.ml
def test_minimum_functionality_clear_sentiment(
    trained_baseline: TfidfLogRegClassifier,
) -> None:
    """Teste comportamental: casos óbvios são classificados corretamente."""
    assert trained_baseline.predict(["adorei recomendo excelente"])[0] == 1
    assert trained_baseline.predict(["horrível detestei péssimo"])[0] == 0
