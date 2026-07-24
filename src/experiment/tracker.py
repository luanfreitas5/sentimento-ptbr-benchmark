"""Rastreamento de experimentos com MLflow.

Envelopa o MLflow em um context manager enxuto, garantindo que cada execução
do benchmark registre parâmetros, métricas e artefatos associados ao domínio de
treino, ao domínio de teste e ao hash dos dados — garantindo linhagem completa.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logger = logging.getLogger(__name__)


class MlflowTracker:
    """Fachada fina sobre o MLflow para o benchmark de sentimento.

    Parameters
    ----------
    tracking_uri : str
        URI do backend do MLflow (ex.: ``./mlruns`` para store local).
    experiment_name : str, optional
        Nome do experimento, by default ``sentimento-ptbr-benchmark``.

    Examples
    --------
    >>> tracker = MlflowTracker("./mlruns")  # doctest: +SKIP
    >>> with tracker.start_run("tfidf_logreg__b2w->olist"):  # doctest: +SKIP
    ...     tracker.log_params({"model": "tfidf_logreg"})
    ...     tracker.log_metrics({"f1_macro": 0.91})
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "sentimento-ptbr-benchmark",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
        """Abre uma execução do MLflow como context manager.

        Parameters
        ----------
        run_name : str
            Nome legível da execução (ex.: ``bertimbau__b2w->olist``).
        tags : dict[str, str] | None, optional
            Tags adicionais (ex.: git SHA, hash dos dados).

        Yields
        ------
        Any
            O objeto de run ativo do MLflow.
        """
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            logger.info("MLflow run iniciada: %s", run_name)
            yield run

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        """Registra um dicionário de parâmetros na run ativa."""
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict[str, float]) -> None:
        """Registra um dicionário de métricas na run ativa."""
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifact(path: Path) -> None:
        """Registra um arquivo (figura, JSON de métricas) como artefato."""
        mlflow.log_artifact(str(path))
