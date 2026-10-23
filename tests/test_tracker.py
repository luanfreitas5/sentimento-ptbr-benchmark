"""Testes do rastreador MLflow (SQLite em diretório temporário)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.tracker import MlflowTracker


@pytest.mark.integration
def test_mlflow_tracker_logs_run(tmp_path: Path) -> None:
    """Uma run registra parâmetros e métricas no backend SQLite sem erro."""
    db_path = tmp_path / "mlruns" / "mlflow.db"
    db_path.parent.mkdir()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MlflowTracker(tracking_uri, experiment_name="teste")

    with tracker.start_run("run_de_teste"):
        tracker.log_params({"model": "tfidf_logreg", "seed": 42})
        tracker.log_metrics({"f1_macro": 0.9})

    # O banco SQLite deve ter sido materializado em disco.
    assert db_path.exists()
