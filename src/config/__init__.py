"""Gestão de configuração, caminhos, logging e reprodutibilidade.

Módulos
-------
paths
    Centraliza todos os caminhos do projeto (``pathlib.Path``), lendo
    ``configs/paths.yaml`` e resolvendo a raiz do repositório.
settings
    Carrega e valida ``config.yaml`` e ``model_params.yaml`` com Pydantic,
    além de variáveis de ambiente/segredos via ``.env``.
logging
    Configura logging com ``RichHandler`` e arquivo rotativo diário.
environment
    Fixa sementes de aleatoriedade (reprodutibilidade determinística).
"""

from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings

__all__ = ["ProjectPaths", "Settings", "get_paths", "load_settings"]
