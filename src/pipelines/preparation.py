"""Pipeline de preparação dos corpora na arquitetura medalhão.

- **bronze -> silver** (``run_bronze_to_silver``): carrega o CSV bronze
  (imutável), limpa, valida o contrato e salva o corpus completo em parquet.
  Registra o hash SHA-256 de cada CSV bronze em ``data/bronze/manifest.json``.
  Também grava ``data/silver/manifest.json`` (linhas e SHA-256 de cada parquet).
- **silver -> gold** (``run_silver_to_gold``): amostra de forma estratificada,
  divide em treino/teste e salva os splits em parquet, com um manifest em
  ``data/gold/manifest.json``.

``run_preparation`` encadeia as duas etapas.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config.logging import CONSOLE
from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings
from src.data.loader import prepare_domain
from src.data.sampler import stratified_sample
from src.data.splitter import train_test_split_domain
from src.exceptions.data import DatasetNotFoundError
from src.schemas.dataset import validate_processed
from src.utils.hashing import hash_file
from src.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def _progress() -> Progress:
    """Cria a barra de progresso rich padrão do projeto."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    )


def split_silver_domain(
    domain: str,
    settings: Settings,
    paths: ProjectPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Lê o silver de um domínio, amostra, valida e divide em treino/teste.

    Parameters
    ----------
    domain : str
        Identificador do domínio.
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Par ``(treino, teste)`` já validado contra o contrato processado.

    Raises
    ------
    DatasetNotFoundError
        Se o parquet silver do domínio não existir.
    """
    silver_path = paths.silver_dataset(domain)
    if not silver_path.exists():
        raise DatasetNotFoundError(
            f"Silver de '{domain}' não encontrado em {silver_path}. "
            "Execute a etapa bronze -> silver antes."
        )

    seed = settings.project.random_seed
    sampling = settings.project.sampling
    split = settings.project.split

    sampled = stratified_sample(
        pl.read_parquet(silver_path),
        sampling.sample_size,
        stratify_by=sampling.stratify_by,
        seed=seed,
    )
    validate_processed(sampled)

    return train_test_split_domain(
        sampled,
        test_size=split.test_size,
        stratify=split.stratify,
        target=settings.project.target,
        seed=seed,
    )


def run_bronze_to_silver(
    domains: Sequence[str] | None = None,
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, int]:
    """Limpa e valida os corpora bronze e persiste a camada silver.

    Parameters
    ----------
    domains : Sequence[str] | None, optional
        Domínios a processar; se ``None``, usa os de ``config.yaml``.
    settings : Settings | None, optional
        Configuração; se ``None``, carrega via ``load_settings()``.
    paths : ProjectPaths | None, optional
        Caminhos; se ``None``, usa ``get_paths()``.

    Returns
    -------
    dict[str, int]
        Mapa ``domínio -> nº de linhas`` do silver.

    Examples
    --------
    >>> run_bronze_to_silver(["b2w"])  # doctest: +SKIP
    """
    settings = settings or load_settings()
    paths = paths or get_paths()
    paths.ensure_output_dirs()
    domains = list(domains or settings.project.datasets)

    rows: dict[str, int] = {}
    bronze_hashes: dict[str, str] = {}
    silver_manifest: dict[str, dict[str, object]] = {}

    with _progress() as progress:
        task = progress.add_task("Bronze -> silver", total=len(domains))
        for domain in domains:
            silver_df = prepare_domain(
                domain, min_chars=settings.project.sampling.min_chars, paths=paths
            )
            validate_processed(silver_df)

            silver_path = paths.silver_dataset(domain)
            silver_path.parent.mkdir(parents=True, exist_ok=True)
            silver_df.write_parquet(silver_path)

            rows[domain] = silver_df.height
            bronze_hashes[domain] = hash_file(paths.bronze_dataset(domain))
            silver_manifest[domain] = {
                "rows": silver_df.height,
                "sha256": hash_file(silver_path),
                "bronze_sha256": bronze_hashes[domain],
            }
            progress.advance(task)

    paths.bronze_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.bronze_dir / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    write_json(previous | bronze_hashes, manifest_path)
    silver_manifest_path = paths.silver_dir / "manifest.json"
    previous_silver = read_json(silver_manifest_path) if silver_manifest_path.exists() else {}
    write_json(previous_silver | silver_manifest, silver_manifest_path)
    logger.info("Silver gerado para %d domínios", len(domains))
    return rows


def run_silver_to_gold(
    domains: Sequence[str] | None = None,
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, dict[str, int]]:
    """Amostra e divide a camada silver, persistindo os splits gold.

    Parameters
    ----------
    domains : Sequence[str] | None, optional
        Domínios a processar; se ``None``, usa os de ``config.yaml``.
    settings : Settings | None, optional
        Configuração; se ``None``, carrega via ``load_settings()``.
    paths : ProjectPaths | None, optional
        Caminhos; se ``None``, usa ``get_paths()``.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapa ``domínio -> {"train": n, "test": n}`` com os tamanhos dos splits.

    Examples
    --------
    >>> run_silver_to_gold(["b2w"])  # doctest: +SKIP
    """
    settings = settings or load_settings()
    paths = paths or get_paths()
    paths.ensure_output_dirs()
    domains = list(domains or settings.project.datasets)

    manifest: dict[str, dict[str, int]] = {}

    with _progress() as progress:
        task = progress.add_task("Silver -> gold", total=len(domains))
        for domain in domains:
            train_df, test_df = split_silver_domain(domain, settings, paths)

            train_path = paths.gold_split(domain, "train")
            test_path = paths.gold_split(domain, "test")
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.write_parquet(train_path)
            test_df.write_parquet(test_path)

            manifest[domain] = {"train": train_df.height, "test": test_df.height}
            progress.advance(task)

    bronze_manifest = paths.bronze_dir / "manifest.json"
    bronze_hashes = read_json(bronze_manifest) if bronze_manifest.exists() else {}
    write_json(
        {"splits": manifest, "bronze_sha256": bronze_hashes},
        paths.gold_dir / "manifest.json",
    )
    logger.info("Gold gerado para %d domínios", len(domains))
    return manifest


def run_preparation(
    domains: Sequence[str] | None = None,
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, dict[str, int]]:
    """Executa bronze -> silver -> gold para os domínios informados.

    Parameters
    ----------
    domains : Sequence[str] | None, optional
        Domínios a preparar; se ``None``, usa os de ``config.yaml``.
    settings : Settings | None, optional
        Configuração; se ``None``, carrega via ``load_settings()``.
    paths : ProjectPaths | None, optional
        Caminhos; se ``None``, usa ``get_paths()``.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapa ``domínio -> {"train": n, "test": n}`` com os tamanhos dos splits gold.

    Examples
    --------
    >>> run_preparation(["b2w"])  # doctest: +SKIP
    """
    settings = settings or load_settings()
    paths = paths or get_paths()
    run_bronze_to_silver(domains, settings, paths)
    return run_silver_to_gold(domains, settings, paths)
