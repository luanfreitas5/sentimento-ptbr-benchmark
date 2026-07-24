"""Exceções customizadas do projeto.

Fornecem mensagens de erro claras em pt-BR e contexto suficiente para depurar
falhas de dados, configuração ou modelagem.

Módulos
-------
base
    Exceção base ``SentiBenchError``.
data
    Erros de dados/contratos (``DatasetNotFoundError``, ``DataValidationError``).
model
    Erros de modelagem (``ModelNotFittedError``, ``MissingDependencyError``).
"""

from src.exceptions.base import SentiBenchError
from src.exceptions.data import DatasetNotFoundError, DataValidationError
from src.exceptions.model import MissingDependencyError, ModelNotFittedError

__all__ = [
    "DataValidationError",
    "DatasetNotFoundError",
    "MissingDependencyError",
    "ModelNotFittedError",
    "SentiBenchError",
]
