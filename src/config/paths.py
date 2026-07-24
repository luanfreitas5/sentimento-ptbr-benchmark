"""Centraliza os caminhos do projeto usando ``pathlib.Path``.

Lê ``configs/paths.yaml`` e resolve todos os caminhos em relação à raiz do
repositório, garantindo que nenhum caminho seja codificado como string ao
longo do código.

Examples
--------
>>> from src.config.paths import get_paths
>>> paths = get_paths()
>>> paths.raw_dir.name
'raw'
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Raiz do repositório: .../sentimento-ptbr-benchmark (dois níveis acima daqui).
ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = ROOT / "configs"
PATHS_CONFIG: Path = CONFIGS_DIR / "paths.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    """Coleção imutável de caminhos usados no projeto.

    Attributes
    ----------
    root : Path
        Raiz do repositório.
    raw_dir, interim_dir, processed_dir : Path
        Estágios dos dados (``data/raw``, ``data/interim``, ``data/processed``).
    models_dir : Path
        Diretório de modelos treinados.
    figures_dir, metrics_dir, model_cards_dir, datasheets_dir : Path
        Saídas de relatório.
    logs_dir : Path
        Diretório de logs.
    mlruns_dir : Path
        Backend store local do MLflow.
    """

    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    metrics_dir: Path
    model_cards_dir: Path
    datasheets_dir: Path
    logs_dir: Path
    mlruns_dir: Path

    def raw_dataset(self, name: str) -> Path:
        """Retorna o caminho do CSV bruto de um domínio (ex.: ``b2w``)."""
        return self.raw_dir / f"{name}.csv"

    def processed_split(self, name: str, split: str) -> Path:
        """Retorna o caminho do parquet de um split (``train``/``test``)."""
        return self.processed_dir / name / f"{split}.parquet"

    def model_dir(self, model_name: str, domain: str) -> Path:
        """Retorna o diretório de um modelo treinado em um domínio."""
        return self.models_dir / model_name / domain

    def ensure_output_dirs(self) -> None:
        """Cria os diretórios de saída (interim, processed, reports, logs...)."""
        for directory in (
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
            self.figures_dir,
            self.metrics_dir,
            self.model_cards_dir,
            self.datasheets_dir,
            self.logs_dir,
            self.mlruns_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_paths(config_file: Path = PATHS_CONFIG) -> ProjectPaths:
    """Carrega e resolve os caminhos do projeto a partir de ``paths.yaml``.

    Parameters
    ----------
    config_file : Path, optional
        Arquivo YAML com os caminhos relativos, by default ``configs/paths.yaml``.

    Returns
    -------
    ProjectPaths
        Estrutura imutável com todos os caminhos resolvidos como ``Path``.

    Raises
    ------
    FileNotFoundError
        Se o arquivo de configuração de caminhos não existir.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.root.exists()
    True
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de caminhos não encontrado: {config_file}")

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    data = cfg["data"]
    reports = cfg["reports"]

    def resolve(rel: str) -> Path:
        return (ROOT / rel).resolve()

    return ProjectPaths(
        root=ROOT,
        raw_dir=resolve(data["raw"]),
        interim_dir=resolve(data["interim"]),
        processed_dir=resolve(data["processed"]),
        models_dir=resolve(cfg["models"]),
        reports_dir=resolve(reports["root"]),
        figures_dir=resolve(reports["figures"]),
        metrics_dir=resolve(reports["metrics"]),
        model_cards_dir=resolve(reports["model_cards"]),
        datasheets_dir=resolve(reports["datasheets"]),
        logs_dir=resolve(cfg["logs"]),
        mlruns_dir=resolve(cfg["mlruns"]),
    )
