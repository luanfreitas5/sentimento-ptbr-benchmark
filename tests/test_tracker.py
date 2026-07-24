"""Testes do rastreador MLflow (store local em diretório temporário)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.tracker import MlflowTracker


@pytest.mark.integration
def test_mlflow_tracker_logs_run(tmp_path: Path) -> None:
    """Uma run registra parâmetros e métricas no store local sem erro."""
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracker = MlflowTracker(tracking_uri, experiment_name="teste")

    with tracker.start_run("run_de_teste"):
        tracker.log_params({"model": "tfidf_logreg", "seed": 42})
        tracker.log_metrics({"f1_macro": 0.9})

    # O store local deve ter sido materializado em disco.
    assert (tmp_path / "mlruns").exists()
