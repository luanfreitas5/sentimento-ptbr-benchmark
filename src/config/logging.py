"""Configura o logging do projeto (console ``rich`` + arquivo rotativo diário).

Todas as mensagens de log são em pt-BR. A configuração é lida de
``configs/logging.yaml`` e pode ser sobrescrita por parâmetros.

Examples
--------
>>> from src.config.logging import configure_logging
>>> logger = configure_logging()
>>> logger.info("Pipeline iniciado")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import yaml
from rich.logging import RichHandler

from src.config.paths import CONFIGS_DIR, get_paths

LOGGING_CONFIG: Path = CONFIGS_DIR / "logging.yaml"


def configure_logging1(config_file: Path = LOGGING_CONFIG) -> logging.Logger:
    """Configura os handlers de console e arquivo a partir do YAML.

    Parameters
    ----------
    config_file : Path, optional
        Caminho do ``logging.yaml``, by default ``configs/logging.yaml``.

    Returns
    -------
    logging.Logger
        Logger raiz configurado.

    Notes
    -----
    - Console: ``RichHandler`` com cores, ``name``/``levelname``/``message``.
    - Arquivo: ``logs/log_YYYY-MM-DD.log`` com rotação diária.
    - Nunca registre PII ou segredos (ver LGPD no CLAUDE.md).

    Examples
    --------
    >>> logger = configure_logging()
    >>> logger.level <= logging.INFO
    True
    """
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # Evita handlers duplicados quando a função é chamada mais de uma vez.
    root.handlers.clear()

    console_cfg = cfg.get("console", {})
    if console_cfg.get("enabled", True):
        console = RichHandler(
            rich_tracebacks=console_cfg.get("rich_tracebacks", True),
            show_path=console_cfg.get("show_path", True),
            markup=True,
        )
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(name)s | %(message)s"))
        root.addHandler(console)

    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", True):
        logs_dir = get_paths().logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Base fixa; o handler acrescenta o sufixo de data na rotação.
        log_path = logs_dir / "log.log"
        file_handler = TimedRotatingFileHandler(
            log_path,
            when=file_cfg.get("when", "midnight"),
            backupCount=file_cfg.get("backup_count", 14),
            encoding=file_cfg.get("encoding", "utf-8"),
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                cfg.get("file_format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"),
                datefmt=cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
        )
        root.addHandler(file_handler)

    return root


def configure_logging(logging_yaml: Path = LOGGING_CONFIG) -> logging.Logger:
    """Configura os handlers de console (rich) e arquivo (rotação diária).

    Parameters
    ----------
    logging_yaml : Path, optional
        Arquivo de configuração de logging, by default ``configs/logging.yaml``.

    Returns
    -------
    logging.Logger
        Logger raiz já configurado.

    Examples
    --------
    >>> logger = configure_logging()
    >>> logger.info("Pipeline iniciado")
    """
    with logging_yaml.open(encoding="utf-8") as handler:
        cfg = yaml.safe_load(handler)

    level = logging.getLevelName(cfg.get("level", "INFO"))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_cfg = cfg.get("console", {})
    console_handler = RichHandler(
        rich_tracebacks=console_cfg.get("rich_tracebacks", True),
        show_path=console_cfg.get("show_path", True),
        show_time=console_cfg.get("show_time", True),
    )
    console_handler.setFormatter(logging.Formatter("%(name)s \t %(message)s"))
    root_logger.addHandler(console_handler)

    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", True):
        logs_dir = get_paths().logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime(file_cfg.get("filename_pattern", "log_%Y-%m-%d.log"))
        file_handler = TimedRotatingFileHandler(
            filename=logs_dir / filename,
            when=file_cfg.get("when", "midnight"),
            backupCount=file_cfg.get("backup_count", 14),
            encoding=file_cfg.get("encoding", "utf-8"),
        )
        file_handler.setFormatter(
            logging.Formatter(
                file_cfg.get("format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s")
            )
        )
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado (atalho para ``logging.getLogger``).

    Parameters
    ----------
    name : str
        Nome do logger, normalmente ``__name__``.

    Returns
    -------
    logging.Logger
        Logger nomeado.
    """
    return logging.getLogger(name)
