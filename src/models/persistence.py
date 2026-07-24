"""Salvamento e carregamento de modelos treinados.

Dispatcher fino que resolve a classe concreta a partir do nome do modelo,
delegando a persistência às próprias implementações (``save``/``load``).
"""

from __future__ import annotations

from pathlib import Path

from src.models.base import SentimentClassifier
from src.models.bertimbau import BertimbauClassifier
from src.models.factory import UnknownModelError
from src.models.tfidf_logreg import TfidfLogRegClassifier


def load_model(name: str, path: Path) -> SentimentClassifier:
    """Carrega um modelo treinado pelo nome e diretório.

    Parameters
    ----------
    name : str
        Identificador do modelo (``tfidf_logreg`` ou ``bertimbau``).
    path : Path
        Diretório onde o modelo foi salvo.

    Returns
    -------
    SentimentClassifier
        Modelo pronto para inferência.

    Raises
    ------
    UnknownModelError
        Se ``name`` não corresponder a nenhum modelo registrado.

    Examples
    --------
    >>> load_model("desconhecido", Path("."))  # doctest: +SKIP
    """
    if name == TfidfLogRegClassifier.name:
        return TfidfLogRegClassifier.load(path)
    if name == BertimbauClassifier.name:
        return BertimbauClassifier.load(path)
    raise UnknownModelError(f"Modelo desconhecido para carregar: '{name}'.")
