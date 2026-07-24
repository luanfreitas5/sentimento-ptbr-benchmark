"""Utilitários de hashing para rastrear a versão de dados e artefatos.

Um hash estável dos dados permite detectar mudanças silenciosas nos corpora e
associar cada modelo aos dados exatos que o produziram.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB por leitura (evita carregar CSVs enormes na RAM)


def hash_file(path: Path) -> str:
    """Retorna o hash SHA-256 de um arquivo, lido em blocos.

    Parameters
    ----------
    path : Path
        Caminho do arquivo a ser hasheado.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> from pathlib import Path
    >>> _ = hash_file(Path("configs/config.yaml"))  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hashing: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """Retorna o hash SHA-256 de uma string (UTF-8).

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.

    Examples
    --------
    >>> len(hash_text("olá"))
    64
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
