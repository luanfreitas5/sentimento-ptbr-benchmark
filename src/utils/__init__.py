"""Funções utilitárias compartilhadas.

Módulos
-------
hashing
    Hash SHA-256 de arquivos e strings para rastrear a versão dos dados.
io
    Leitura/escrita de artefatos JSON de métricas.
"""

from src.utils.hashing import hash_file, hash_text
from src.utils.io import read_json, write_json

__all__ = ["hash_file", "hash_text", "read_json", "write_json"]
