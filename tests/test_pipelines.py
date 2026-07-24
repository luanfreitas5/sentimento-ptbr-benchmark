"""Teste de integração do pipeline ponta a ponta (baseline, sem torch).

Exercita ``run_preparation`` e ``run_benchmark`` sobre dois domínios sintéticos
minúsculos, cobrindo a orquestração sem depender do extra ``bert``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

from src.config.paths import ProjectPaths
from src.config.settings import (
    EvaluationConfig,
    ModelParamsConfig,
    ProjectConfig,
    SamplingConfig,
    Settings,
    SplitConfig,
)
from src.pipelines.benchmark import run_benchmark
from src.pipelines.preparation import run_preparation
from src.utils.io import read_json
from tests.conftest import make_test_config

_DOMAINS = ["b2w", "olist"]


def _build_paths(tmp_path: Path) -> ProjectPaths:
    """ProjectPaths inteiramente sob um diretório temporário."""
    return ProjectPaths(
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        metrics_dir=tmp_path / "metrics",
        model_cards_dir=tmp_path / "model_cards",
        datasheets_dir=tmp_path / "datasheets",
        logs_dir=tmp_path / "logs",
        mlruns_dir=tmp_path / "mlruns",
    )


def _build_settings() -> Settings:
    """Settings com dois domínios e baseline adequado a dados minúsculos."""
    return Settings(
        project=ProjectConfig(
            datasets=_DOMAINS,
            sampling=SamplingConfig(sample_size=None, min_chars=3),
            split=SplitConfig(test_size=0.3, stratify=True),
        ),
        model_params=ModelParamsConfig(
            tfidf_logreg=make_test_config(),
            evaluation=EvaluationConfig(bootstrap_samples=100),
        ),
    )


def _write_raw(paths: ProjectPaths) -> None:
    """Escreve CSVs brutos sintéticos com sinal claro para os dois domínios."""
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    positives = [
        "produto excelente adorei recomendo muito bom",
        "ótima qualidade entrega rápida perfeito top",
        "maravilhoso superou expectativas amei demais",
    ] * 6
    negatives = [
        "péssimo produto quebrado detestei horrível",
        "muito ruim veio com defeito decepcionante lixo",
        "atendimento terrível arrependido não recomendo",
    ] * 6
    texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    for domain in _DOMAINS:
        pl.DataFrame(
            {
                "review_text": texts,
                "polarity": [float(v) for v in labels],
                "rating": [5.0 if v else 1.0 for v in labels],
            }
        ).write_csv(paths.raw_dataset(domain))


@pytest.mark.integration
def test_end_to_end_baseline_benchmark(tmp_path: Path) -> None:
    """Prepara os corpora e roda o benchmark cross-dataset do baseline."""
    paths = _build_paths(tmp_path)
    settings = _build_settings()
    _write_raw(paths)

    manifest = run_preparation(domains=_DOMAINS, settings=settings, paths=paths)
    assert set(manifest) == set(_DOMAINS)
    assert paths.processed_split("b2w", "train").exists()

    results = run_benchmark(
        ["tfidf_logreg"],
        domains=_DOMAINS,
        settings=settings,
        paths=paths,
        track=False,
    )

    result = results["tfidf_logreg"]
    assert len(result.f1_matrix) == 2
    assert len(result.f1_matrix[0]) == 2

    payload = read_json(paths.metrics_dir / "benchmark_tfidf_logreg.json")
    assert payload["domains"] == _DOMAINS
    assert (paths.figures_dir / "cross_dataset_tfidf_logreg.png").exists()
