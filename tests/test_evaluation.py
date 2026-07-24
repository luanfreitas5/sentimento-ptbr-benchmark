"""Testes do avaliador, do teste de McNemar e do resumo do benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.paths import ProjectPaths
from src.evaluation.evaluator import evaluate_model
from src.evaluation.reports import _in_out_domain_means, build_benchmark_summary
from src.evaluation.significance import mcnemar_test
from src.models.tfidf_logreg import TfidfLogRegClassifier
from src.utils.io import write_json


@pytest.mark.smoke
def test_evaluate_model_returns_result(
    trained_baseline: TfidfLogRegClassifier,
) -> None:
    """A avaliação devolve métrica headline e IC coerentes."""
    result = evaluate_model(
        trained_baseline,
        ["produto bom", "produto ruim"],
        [1, 0],
        bootstrap_samples=50,
    )
    assert 0.0 <= result.headline <= 1.0
    assert result.ci_lower <= result.ci_upper
    assert len(result.confusion) == 2


def test_mcnemar_identical_predictions_not_significant() -> None:
    """Modelos idênticos não têm diferença significativa."""
    y_true = [0, 1, 0, 1, 1, 0]
    result = mcnemar_test(y_true, y_true, y_true)
    assert result.significant is False
    assert result.n01 == 0
    assert result.n10 == 0


def test_mcnemar_counts_disagreements() -> None:
    """As discordâncias pareadas são contadas corretamente."""
    y_true = [1, 1, 1, 1]
    pred_a = [1, 1, 0, 0]  # acerta 2
    pred_b = [1, 0, 1, 1]  # acerta 3
    result = mcnemar_test(y_true, pred_a, pred_b)
    # B acerta sozinho nos índices 2 e 3; A acerta sozinho no índice 1.
    assert result.n01 == 2
    assert result.n10 == 1


def test_in_out_domain_means() -> None:
    """Médias in-domain (diagonal) e out-domain (fora) são separadas."""
    in_domain, out_domain = _in_out_domain_means([[0.9, 0.7], [0.6, 0.8]])
    assert in_domain == pytest.approx(0.85)
    assert out_domain == pytest.approx(0.65)


def test_build_benchmark_summary_computes_gap(tmp_path: Path) -> None:
    """O resumo calcula o gap de generalização a partir do JSON salvo."""
    paths = ProjectPaths(
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
    write_json(
        {"f1_macro_matrix": [[0.9, 0.7], [0.6, 0.8]]},
        paths.metrics_dir / "benchmark_tfidf_logreg.json",
    )
    summary = build_benchmark_summary(["tfidf_logreg"], paths=paths)
    assert summary["tfidf_logreg"]["generalization_gap"] == pytest.approx(0.2)
