"""Pipelines ponta a ponta do benchmark.

Módulos
-------
preparation
    Prepara os corpora na arquitetura medalhão (bronze -> silver -> gold):
    limpa, valida, amostra, divide e salva os splits.
benchmark
    Executa o benchmark cross-dataset (treina em N domínios, testa em N),
    registra no MLflow e gera métricas/figuras.
"""

from src.pipelines.benchmark import run_benchmark
from src.pipelines.preparation import (
    run_bronze_to_silver,
    run_preparation,
    run_silver_to_gold,
)

__all__ = ["run_benchmark", "run_bronze_to_silver", "run_preparation", "run_silver_to_gold"]
