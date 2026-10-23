"""Centraliza os caminhos do projeto usando ``pathlib.Path``.

Lê ``configs/paths.yaml`` e resolve todos os caminhos em relação à raiz do
repositório, garantindo que nenhum caminho seja codificado como string ao
longo do código.

Examples
--------
>>> from src.config.paths import get_paths
>>> paths = get_paths()
>>> paths.bronze_dir.name
'bronze'
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
    bronze_dir, silver_dir, gold_dir : Path
        Camadas medalhão dos dados: ``data/bronze`` (como recebido, imutável),
        ``data/silver`` (limpo, tipado e validado) e ``data/gold`` (splits
        prontos para modelagem).
    models_dir : Path
        Diretório de modelos treinados.
    figures_dir, metrics_dir, model_cards_dir, datasheets_dir : Path
        Saídas de relatório.
    logs_dir : Path
        Diretório de logs.
    mlruns_dir : Path
        Diretório do banco SQLite do MLflow (``mlruns/mlflow.db``).
    """

    root: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    metrics_dir: Path
    model_cards_dir: Path
    datasheets_dir: Path
    logs_dir: Path
    mlruns_dir: Path

    def bronze_dataset(self, name: str) -> Path:
        """Retorna o caminho do CSV bronze de um domínio (ex.: ``b2w``)."""
        return self.bronze_dir / f"{name}.csv"

    def silver_dataset(self, name: str) -> Path:
        """Retorna o caminho do parquet silver (limpo e validado) de um domínio."""
        return self.silver_dir / f"{name}.parquet"

    def gold_split(self, name: str, split: str) -> Path:
        """Retorna o caminho do parquet gold de um split (``train``/``test``)."""
        return self.gold_dir / name / f"{split}.parquet"

    def model_dir(self, model_name: str, domain: str) -> Path:
        """Retorna o diretório de um modelo treinado em um domínio."""
        return self.models_dir / model_name / domain

    def ensure_output_dirs(self) -> None:
        """Cria os diretórios de saída (silver, gold, reports, logs...)."""
        for directory in (
            self.silver_dir,
            self.gold_dir,
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
        bronze_dir=resolve(data["bronze"]),
        silver_dir=resolve(data["silver"]),
        gold_dir=resolve(data["gold"]),
        models_dir=resolve(cfg["models"]),
        reports_dir=resolve(reports["root"]),
        figures_dir=resolve(reports["figures"]),
        metrics_dir=resolve(reports["metrics"]),
        model_cards_dir=resolve(reports["model_cards"]),
        datasheets_dir=resolve(reports["datasheets"]),
        logs_dir=resolve(cfg["logs"]),
        mlruns_dir=resolve(cfg["mlruns"]),
    )
