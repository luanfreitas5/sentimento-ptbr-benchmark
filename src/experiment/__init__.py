"""Rastreamento de experimentos com MLflow.

Módulos
-------
tracker
    Context manager fino sobre o MLflow para logar parâmetros, métricas e
    artefatos de cada execução do benchmark.
"""

from src.experiment.tracker import MlflowTracker

__all__ = ["MlflowTracker"]
