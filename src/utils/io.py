"""Leitura e escrita de artefatos JSON (métricas, manifests).

Centraliza a serialização de resultados numéricos garantindo criação de
diretórios e codificação UTF-8 (acentos preservados).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: Path, *, indent: int = 2) -> None:
    """Escreve um dicionário em JSON, criando diretórios pai se preciso.

    Parameters
    ----------
    data : dict[str, Any]
        Conteúdo serializável a ser gravado.
    path : Path
        Caminho de destino do arquivo ``.json``.
    indent : int, optional
        Nível de indentação, by default 2.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_json({"f1": 0.9}, Path("reports/metrics/exemplo.json"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.json``.

    Returns
    -------
    dict[str, Any]
        Conteúdo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
