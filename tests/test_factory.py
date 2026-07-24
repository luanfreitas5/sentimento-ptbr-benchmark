"""Testes da fábrica de modelos."""

from __future__ import annotations

import pytest

from src.config.settings import ModelParamsConfig
from src.models.bertimbau import BertimbauClassifier
from src.models.factory import UnknownModelError, build_model, list_models
from src.models.tfidf_logreg import TfidfLogRegClassifier


@pytest.mark.smoke
def test_list_models_contains_both() -> None:
    """Ambos os modelos estão registrados."""
    assert set(list_models()) == {"tfidf_logreg", "bertimbau"}


def test_build_baseline() -> None:
    """A fábrica cria o baseline pelo nome."""
    model = build_model("tfidf_logreg", ModelParamsConfig())
    assert isinstance(model, TfidfLogRegClassifier)


def test_build_bertimbau_instance_is_lazy() -> None:
    """Instanciar o BERTimbau não exige torch (imports preguiçosos)."""
    model = build_model("bertimbau", ModelParamsConfig())
    assert isinstance(model, BertimbauClassifier)


def test_build_unknown_model_raises() -> None:
    """Um nome desconhecido levanta ``UnknownModelError``."""
    with pytest.raises(UnknownModelError):
        build_model("inexistente", ModelParamsConfig())
