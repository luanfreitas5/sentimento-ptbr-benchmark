"""Normalização leve de texto em PT-BR para o baseline TF-IDF.

A limpeza é deliberadamente conservadora: acentos são preservados (carregam
sinal em português) e a negação não é removida. URLs, e-mails, menções e
repetições de pontuação/caracteres — que só adicionam ruído ao TF-IDF — são
neutralizados.

O BERTimbau **não** usa esta função: seu tokenizer WordPiece foi treinado
sobre texto cru e lida melhor com o original.
"""

from __future__ import annotations

import re

# Padrões pré-compilados (custo único de compilação).
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
# Colapsa 3+ repetições do mesmo caractere para no máximo 2 (ex.: "ameeeei").
_CHAR_REPEAT_RE = re.compile(r"(.)\1{2,}")
# Colapsa repetições de pontuação (ex.: "!!!" -> "!").
_PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1+")


def normalize_whitespace(text: str) -> str:
    """Colapsa espaços em branco consecutivos e remove bordas.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    str
        Texto com espaçamento normalizado.

    Examples
    --------
    >>> normalize_whitespace("  ótimo   produto \\n ")
    'ótimo produto'
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    """Normaliza um texto de review para o baseline TF-IDF.

    Aplica, em ordem: minúsculas, remoção de URLs/e-mails/menções, colapso de
    caracteres e pontuação repetidos e normalização de espaços. Acentos são
    preservados.

    Parameters
    ----------
    text : str | None
        Texto original da review; ``None`` é tratado como string vazia.

    Returns
    -------
    str
        Texto limpo (pode ser vazio se a entrada só tinha ruído).

    Examples
    --------
    >>> clean_text("Ameeeei!!! Visite http://loja.com  :)")
    'ameei! visite :)'
    >>> clean_text(None)
    ''
    """
    if not text:
        return ""

    lowered = text.lower()
    lowered = _URL_RE.sub(" ", lowered)
    lowered = _EMAIL_RE.sub(" ", lowered)
    lowered = _MENTION_RE.sub(" ", lowered)
    lowered = _CHAR_REPEAT_RE.sub(r"\1\1", lowered)
    lowered = _PUNCT_REPEAT_RE.sub(r"\1", lowered)
    return normalize_whitespace(lowered)
