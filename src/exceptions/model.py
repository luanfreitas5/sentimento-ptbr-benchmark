"""Exceções relacionadas a modelagem."""

from __future__ import annotations

from src.exceptions.base import SentiBenchError


class ModelNotFittedError(SentiBenchError):
    """Levantada ao prever/salvar um modelo que ainda não foi treinado."""


class MissingDependencyError(SentiBenchError):
    """Levantada quando uma dependência opcional necessária está ausente.

    Usada, por exemplo, quando o BERTimbau é solicitado sem o extra ``bert``
    (torch/transformers) instalado.
    """
