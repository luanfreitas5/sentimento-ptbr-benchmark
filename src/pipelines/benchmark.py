"""Pipeline do benchmark cross-dataset — o coração do projeto.

Para cada modelo e cada domínio de treino, treina uma vez e avalia em **todos**
os domínios de teste, montando a matriz cross-dataset de F1-macro. A diagonal é
o desempenho in-domain; o restante mede a transferência entre domínios. Quando
os dois modelos estão presentes, compara-os in-domain com o teste de McNemar.

Registra tudo no MLflow (parâmetros, métricas, figuras) e persiste as matrizes
em JSON e como heatmaps.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from models.tfidf_logreg import TfidfLogRegClassifier
from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings
from src.constants.columns import TEXT_CLEAN, RawColumns
from src.evaluation.evaluator import EvaluationResult, evaluate_model
from src.evaluation.significance import mcnemar_test
from src.experiment.tracker import MlflowTracker
from src.models.bertimbau import BertimbauClassifier
from src.models.factory import build_model
from src.utils.io import write_json
from src.visualization.benchmark import plot_cross_dataset_matrix
from src.visualization.theme import apply_theme, save_figure

logger = logging.getLogger(__name__)


@dataclass
class DomainSplits:
    """Textos e rótulos de treino/teste de um domínio, já em memória."""

    train_text_clean: list[str]
    train_text_raw: list[str]
    train_labels: list[int]
    test_text_clean: list[str]
    test_text_raw: list[str]
    test_labels: list[int]


@dataclass
class BenchmarkResult:
    """Resultados consolidados do benchmark de um modelo.

    Attributes
    ----------
    model_name : str
        Identificador do modelo.
    domains : list[str]
        Domínios na ordem das linhas/colunas da matriz.
    f1_matrix : list[list[float]]
        Matriz ``N x N`` de F1-macro (linha = treino, coluna = teste).
    cells : dict[str, dict]
        Detalhe por célula ``"train->test"`` (todas as métricas + IC).
    diagonal_predictions : dict[str, np.ndarray]
        Predições in-domain por domínio (para o teste de McNemar).
    """

    model_name: str
    domains: list[str]
    f1_matrix: list[list[float]]
    cells: dict[str, dict] = field(default_factory=dict)
    diagonal_predictions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def text_column_for(model_name: str) -> str:
    """Escolhe a coluna de texto adequada a cada modelo.

    O baseline TF-IDF usa o texto limpo; o BERTimbau usa o texto original
    (seu tokenizer lida melhor com o texto cru).

    Parameters
    ----------
    model_name : str
        Identificador do modelo.

    Returns
    -------
    str
        Nome da coluna de texto a usar.

    Examples
    --------
    >>> text_column_for("bertimbau")
    'review_text'
    >>> text_column_for("tfidf_logreg")
    'text_clean'
    """
    return RawColumns.REVIEW_TEXT if model_name == BertimbauClassifier.name else TEXT_CLEAN


def load_domain_splits(domain: str, paths: ProjectPaths) -> DomainSplits:
    """Carrega os splits processados de um domínio para a memória.

    Parameters
    ----------
    domain : str
        Identificador do domínio.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    DomainSplits
        Textos (limpo e original) e rótulos de treino e teste.
    """
    train = pl.read_parquet(paths.processed_split(domain, "train"))
    test = pl.read_parquet(paths.processed_split(domain, "test"))
    return DomainSplits(
        train_text_clean=train[TEXT_CLEAN].to_list(),
        train_text_raw=train[RawColumns.REVIEW_TEXT].to_list(),
        train_labels=train[RawColumns.POLARITY].to_list(),
        test_text_clean=test[TEXT_CLEAN].to_list(),
        test_text_raw=test[RawColumns.REVIEW_TEXT].to_list(),
        test_labels=test[RawColumns.POLARITY].to_list(),
    )


def _train_texts(splits: DomainSplits, model_name: str) -> list[str]:
    """Seleciona os textos de treino conforme o modelo."""
    if model_name == BertimbauClassifier.name:
        return splits.train_text_raw
    return splits.train_text_clean


def _test_texts(splits: DomainSplits, model_name: str) -> list[str]:
    """Seleciona os textos de teste conforme o modelo."""
    if model_name == BertimbauClassifier.name:
        return splits.test_text_raw
    return splits.test_text_clean


def _benchmark_single_model(
    model_name: str,
    domains: Sequence[str],
    data: dict[str, DomainSplits],
    settings: Settings,
    tracker: MlflowTracker | None,
) -> BenchmarkResult:
    """Roda o benchmark cross-dataset de um único modelo."""
    eval_cfg = settings.model_params.evaluation
    seed = settings.project.random_seed
    n = len(domains)
    f1_matrix = [[0.0] * n for _ in range(n)]
    result = BenchmarkResult(model_name=model_name, domains=list(domains), f1_matrix=f1_matrix)

    for i, train_domain in enumerate(domains):
        logger.info("[%s] treinando no domínio '%s'", model_name, train_domain)
        model = build_model(model_name, settings.model_params, seed=seed)
        model.fit(_train_texts(data[train_domain], model_name), data[train_domain].train_labels)

        for j, test_domain in enumerate(domains):
            splits = data[test_domain]
            evaluation = evaluate_model(
                model,
                _test_texts(splits, model_name),
                splits.test_labels,
                primary_metric=eval_cfg.primary_metric,
                bootstrap_samples=eval_cfg.bootstrap_samples,
                confidence=eval_cfg.confidence_level,
                seed=seed,
            )
            f1_matrix[i][j] = round(evaluation.headline, 6)
            cell_key = f"{train_domain}->{test_domain}"
            result.cells[cell_key] = {
                "train": train_domain,
                "test": test_domain,
                "in_domain": train_domain == test_domain,
                "metrics": evaluation.metrics,
                "ci_lower": evaluation.ci_lower,
                "ci_upper": evaluation.ci_upper,
                "confusion": evaluation.confusion,
            }
            if train_domain == test_domain:
                result.diagonal_predictions[train_domain] = evaluation.y_pred

            if tracker is not None:
                _log_cell(tracker, model_name, cell_key, evaluation, seed)

    return result


def _log_cell(
    tracker: MlflowTracker,
    model_name: str,
    cell_key: str,
    evaluation: EvaluationResult,
    seed: int,
) -> None:
    """Registra uma célula do benchmark como uma run do MLflow."""
    with tracker.start_run(f"{model_name}__{cell_key}"):
        tracker.log_params({"model": model_name, "pair": cell_key, "seed": seed})
        tracker.log_metrics(evaluation.metrics)


def _save_outputs(result: BenchmarkResult, paths: ProjectPaths) -> None:
    """Salva a matriz (JSON) e o heatmap cross-dataset de um modelo."""
    write_json(
        {
            "model": result.model_name,
            "domains": result.domains,
            "f1_macro_matrix": result.f1_matrix,
            "cells": result.cells,
        },
        paths.metrics_dir / f"benchmark_{result.model_name}.json",
    )
    fig = plot_cross_dataset_matrix(
        result.f1_matrix,
        result.domains,
        model_name=result.model_name,
    )
    save_figure(fig, paths.figures_dir / f"cross_dataset_{result.model_name}.png")


def _compare_models_in_domain(
    results: dict[str, BenchmarkResult],
    data: dict[str, DomainSplits],
    domains: Sequence[str],
    paths: ProjectPaths,
) -> None:
    """Compara os dois modelos in-domain (diagonal) via teste de McNemar."""

    baseline_name = TfidfLogRegClassifier.name
    bert_name = BertimbauClassifier.name
    if baseline_name not in results or bert_name not in results:
        return

    comparison: dict[str, dict] = {}
    for domain in domains:
        pred_a = results[baseline_name].diagonal_predictions.get(domain)
        pred_b = results[bert_name].diagonal_predictions.get(domain)
        if pred_a is None or pred_b is None:
            continue
        test = mcnemar_test(data[domain].test_labels, pred_a, pred_b)
        comparison[domain] = {
            "statistic": test.statistic,
            "p_value": test.p_value,
            "baseline_only_correct": test.n10,
            "bertimbau_only_correct": test.n01,
            "significant": test.significant,
        }
    write_json(comparison, paths.metrics_dir / "significance_mcnemar.json")
    logger.info("Comparação de significância (McNemar) salva para %d domínios", len(comparison))


def run_benchmark(
    models: Sequence[str],
    domains: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
    track: bool = True,
) -> dict[str, BenchmarkResult]:
    """Executa o benchmark cross-dataset completo.

    Parameters
    ----------
    models : Sequence[str]
        Modelos a avaliar (``tfidf_logreg`` e/ou ``bertimbau``).
    domains : Sequence[str] | None, optional
        Domínios; se ``None``, usa os de ``config.yaml``.
    settings : Settings | None, optional
        Configuração; se ``None``, carrega via ``load_settings()``.
    paths : ProjectPaths | None, optional
        Caminhos; se ``None``, usa ``get_paths()``.
    track : bool, optional
        Se ``True``, registra as execuções no MLflow, by default True.

    Returns
    -------
    dict[str, BenchmarkResult]
        Resultados por modelo (matriz F1, células e predições in-domain).

    Examples
    --------
    >>> run_benchmark(["tfidf_logreg"], ["b2w", "olist"], track=False)  # doctest: +SKIP
    """
    settings = settings or load_settings()
    paths = paths or get_paths()
    paths.ensure_output_dirs()
    apply_theme()
    domains = list(domains or settings.project.datasets)

    logger.info("Carregando splits de %d domínios", len(domains))
    data = {domain: load_domain_splits(domain, paths) for domain in domains}

    tracker = (
        MlflowTracker(settings.mlflow_tracking_uri, settings.project.project_name)
        if track
        else None
    )

    results: dict[str, BenchmarkResult] = {}
    for model_name in models:
        result = _benchmark_single_model(model_name, domains, data, settings, tracker)
        _save_outputs(result, paths)
        results[model_name] = result

    _compare_models_in_domain(results, data, domains, paths)
    logger.info("Benchmark concluído para modelos: %s", ", ".join(models))
    return results
