"""Pipeline de preparação dos corpora (raw -> processed).

Para cada domínio: carrega, limpa, valida o contrato, amostra de forma
estratificada, divide em treino/teste e salva em parquet. Escreve também um
manifest com o hash dos arquivos brutos para rastrear a versão dos dados.
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

from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings
from src.data.loader import prepare_domain
from src.data.sampler import stratified_sample
from src.data.splitter import train_test_split_domain
from src.schemas.dataset import validate_processed
from src.utils.hashing import hash_file
from src.utils.io import write_json

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
    )


def prepare_and_split_domain(
    domain: str,
    settings: Settings,
    paths: ProjectPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prepara, amostra, valida e divide um único domínio.

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
    """
    seed = settings.project.random_seed
    sampling = settings.project.sampling
    split = settings.project.split

    prepared = prepare_domain(domain, min_chars=sampling.min_chars, paths=paths)
    sampled = stratified_sample(
        prepared,
        sampling.sample_size,
        stratify_by=sampling.stratify_by,
        seed=seed,
    )
    validate_processed(sampled)

    train_df, test_df = train_test_split_domain(
        sampled,
        test_size=split.test_size,
        stratify=split.stratify,
        target=settings.project.target,
        seed=seed,
    )
    return train_df, test_df


def run_preparation(
    domains: Sequence[str] | None = None,
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, dict[str, int]]:
    """Executa a preparação de todos os domínios e persiste os splits.

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
        Mapa ``domínio -> {"train": n, "test": n}`` com os tamanhos dos splits.

    Examples
    --------
    >>> run_preparation(["b2w"])  # doctest: +SKIP
    """
    settings = settings or load_settings()
    paths = paths or get_paths()
    paths.ensure_output_dirs()
    domains = list(domains or settings.project.datasets)

    manifest: dict[str, dict[str, int]] = {}
    data_hashes: dict[str, str] = {}

    with _progress() as progress:
        task = progress.add_task("Preparando corpora", total=len(domains))
        for domain in domains:
            train_df, test_df = prepare_and_split_domain(domain, settings, paths)

            train_path = paths.processed_split(domain, "train")
            test_path = paths.processed_split(domain, "test")
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.write_parquet(train_path)
            test_df.write_parquet(test_path)

            manifest[domain] = {"train": train_df.height, "test": test_df.height}
            data_hashes[domain] = hash_file(paths.raw_dataset(domain))
            progress.advance(task)

    write_json(
        {"splits": manifest, "raw_sha256": data_hashes},
        paths.processed_dir / "manifest.json",
    )
    logger.info("Preparação concluída para %d domínios", len(domains))
    return manifest
