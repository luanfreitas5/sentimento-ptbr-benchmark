"""Fábrica de modelos: cria instâncias a partir do nome e da configuração.

Aplica o padrão *Factory* para manter o pipeline de benchmark agnóstico às
implementações concretas — adicionar um novo modelo é registrar uma entrada
aqui, sem tocar no orquestrador.
"""

from __future__ import annotations

from src.config.settings import ModelParamsConfig
from src.exceptions.base import SentiBenchError
from src.models.base import SentimentClassifier
from src.models.bertimbau import BertimbauClassifier
from src.models.tfidf_logreg import TfidfLogRegClassifier

_MODEL_NAMES = (TfidfLogRegClassifier.name, BertimbauClassifier.name)


class UnknownModelError(SentiBenchError):
    """Levantada quando um nome de modelo desconhecido é solicitado."""


def list_models() -> tuple[str, ...]:
    """Lista os identificadores de modelo disponíveis.

    Returns
    -------
    tuple[str, ...]
        Nomes registrados (ex.: ``('tfidf_logreg', 'bertimbau')``).

    Examples
    --------
    >>> "tfidf_logreg" in list_models()
    True
    """
    return _MODEL_NAMES


def build_model(
    name: str,
    params: ModelParamsConfig,
    *,
    seed: int = 42,
) -> SentimentClassifier:
    """Cria uma instância de modelo (não treinada) pelo nome.

    Parameters
    ----------
    name : str
        Identificador do modelo (``tfidf_logreg`` ou ``bertimbau``).
    params : ModelParamsConfig
        Configuração validada com os hiperparâmetros de ambos os modelos.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    SentimentClassifier
        Instância pronta para ``fit``.

    Raises
    ------
    UnknownModelError
        Se ``name`` não corresponder a nenhum modelo registrado.

    Examples
    --------
    >>> from src.config.settings import ModelParamsConfig
    >>> model = build_model("tfidf_logreg", ModelParamsConfig())
    >>> model.name
    'tfidf_logreg'
    """
    if name == TfidfLogRegClassifier.name:
        return TfidfLogRegClassifier(params.tfidf_logreg, seed=seed)
    if name == BertimbauClassifier.name:
        return BertimbauClassifier(params.bertimbau, seed=seed)
    raise UnknownModelError(
        f"Modelo desconhecido: '{name}'. Disponíveis: {', '.join(_MODEL_NAMES)}."
    )
