"""Limpeza e normalização de texto em PT-BR.

Módulos
-------
text
    Normalização leve de texto para o baseline TF-IDF (minúsculas, remoção de
    URLs/ruído, colapso de espaços). O BERTimbau usa o texto original, pois seu
    tokenizer WordPiece lida melhor com o texto cru.
"""

from src.preprocessing.text import clean_text, normalize_whitespace

__all__ = ["clean_text", "normalize_whitespace"]
