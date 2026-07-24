"""Exceção base do projeto."""

from __future__ import annotations


class SentiBenchError(Exception):
    """Exceção base para todos os erros do benchmark de sentimento.

    Todas as exceções customizadas do projeto herdam desta, permitindo
    capturar qualquer erro do domínio com um único ``except``.
    """
