"""Testes de integração da carga e preparação de um domínio."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.config.paths import ProjectPaths
from src.data.loader import prepare_domain
from src.exceptions.data import DatasetNotFoundError


def _paths_with_bronze(tmp_path: Path) -> ProjectPaths:
    """Constrói ``ProjectPaths`` apontando o bronze para um diretório temporário."""
    return ProjectPaths(
        root=tmp_path,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        metrics_dir=tmp_path / "metrics",
        model_cards_dir=tmp_path / "model_cards",
        datasheets_dir=tmp_path / "datasheets",
        logs_dir=tmp_path / "logs",
        mlruns_dir=tmp_path / "mlruns",
    )


def test_prepare_domain_cleans_and_annotates(tmp_path: Path) -> None:
    """Prepara um CSV sintético: remove nulos/curtos e anota o domínio."""
    paths = _paths_with_bronze(tmp_path)
    paths.bronze_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "review_text": ["Ótimo produto!", "Ruim demais", None, "ok"],
            "polarity": [1.0, 0.0, 1.0, 1.0],
            "rating": [5.0, 1.0, 4.0, 3.0],
        }
    ).write_csv(paths.bronze_dataset("b2w"))

    prepared = prepare_domain("b2w", min_chars=3, paths=paths)

    assert "text_clean" in prepared.columns
    assert (prepared["dataset"] == "b2w").all()
    assert prepared["review_text"].null_count() == 0
    # "ok" (2 chars) é descartado por min_chars=3; a linha nula também.
    assert prepared.height == 2


def test_prepare_domain_missing_file_raises(tmp_path: Path) -> None:
    """Um domínio inexistente levanta ``DatasetNotFoundError``."""
    paths = _paths_with_bronze(tmp_path)
    paths.bronze_dir.mkdir(parents=True)
    with pytest.raises(DatasetNotFoundError):
        prepare_domain("inexistente", paths=paths)
