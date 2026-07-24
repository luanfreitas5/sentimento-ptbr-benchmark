"""Avaliação rigorosa: métricas, incerteza e testes de significância.

Módulos
-------
evaluator
    Avalia um modelo em um conjunto de teste, retornando métricas, IC e
    matriz de confusão.
significance
    Teste de McNemar para comparar dois classificadores no mesmo teste.
reports
    Consolida as matrizes cross-dataset em um resumo in/out-of-domain.
"""

from src.evaluation.evaluator import EvaluationResult, evaluate_model
from src.evaluation.reports import build_benchmark_summary
from src.evaluation.significance import mcnemar_test

__all__ = [
    "EvaluationResult",
    "build_benchmark_summary",
    "evaluate_model",
    "mcnemar_test",
]
