"""Exceções relacionadas a dados e contratos."""

from __future__ import annotations

from src.exceptions.base import SentiBenchError


class DatasetNotFoundError(SentiBenchError):
    """Levantada quando um corpus esperado não é encontrado em disco."""


class DataValidationError(SentiBenchError):
    """Levantada quando um DataFrame viola o contrato de dados (pandera)."""
