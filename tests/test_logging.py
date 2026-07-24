"""Testes da configuração de logging."""

from __future__ import annotations

import logging

from src.config.logging import configure_logging, get_logger


def test_configure_logging_returns_root_with_handlers() -> None:
    """Configurar o logging registra ao menos um handler no logger raiz."""
    root = configure_logging()
    assert isinstance(root, logging.Logger)
    assert root.handlers


def test_get_logger_named() -> None:
    """``get_logger`` devolve um logger com o nome informado."""
    assert get_logger("sentibench.teste").name == "sentibench.teste"
