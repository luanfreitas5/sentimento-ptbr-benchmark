"""Pipelines ponta a ponta do benchmark.

Módulos
-------
preparation
    Prepara os corpora: limpa, valida, amostra, divide e salva os splits.
benchmark
    Executa o benchmark cross-dataset (treina em N domínios, testa em N),
    registra no MLflow e gera métricas/figuras.
"""

from src.pipelines.benchmark import run_benchmark
from src.pipelines.preparation import run_preparation

__all__ = ["run_benchmark", "run_preparation"]
