"""Testes da normalização de texto (unitários + baseados em propriedades)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.preprocessing.text import clean_text, normalize_whitespace


@pytest.mark.smoke
def test_clean_text_lowercases_and_collapses_repeats() -> None:
    """Minúsculas, colapso de caracteres e pontuação repetidos."""
    assert clean_text("Ameeeei!!!") == "ameei!"


@pytest.mark.smoke
def test_clean_text_removes_urls_and_mentions() -> None:
    """URLs e menções viram espaço e são removidas do resultado."""
    result = clean_text("veja http://loja.com e @perfil")
    assert "http" not in result
    assert "@perfil" not in result


def test_clean_text_none_returns_empty() -> None:
    """Entrada ``None`` é tratada como string vazia."""
    assert clean_text(None) == ""


def test_clean_text_preserves_accents() -> None:
    """Acentos carregam sinal em PT-BR e devem ser preservados."""
    assert "ótimo" in clean_text("Ótimo")


def test_normalize_whitespace_collapses_spaces() -> None:
    """Espaços consecutivos e bordas são normalizados."""
    assert normalize_whitespace("  a   b \n c ") == "a b c"


@given(st.text())
def test_clean_text_never_raises_and_returns_str(text: str) -> None:
    """Invariante: para qualquer texto, não levanta e retorna string."""
    result = clean_text(text)
    assert isinstance(result, str)


@given(st.text())
def test_clean_text_has_no_double_ascii_space(text: str) -> None:
    """Invariante: runs de espaços ASCII são sempre colapsados."""
    assert "  " not in clean_text(text)
