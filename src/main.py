"""Orquestração principal do benchmark de sentimento em PT-BR (CLI).

Expõe subcomandos que espelham as etapas do pipeline (``--log-level`` ajusta o log):

- ``prepare``  : limpa, valida, amostra e divide os corpora (bronze -> silver -> gold).
- ``bronze-to-silver`` / ``silver-to-gold`` : executa cada camada isoladamente.
- ``train``    : roda o benchmark cross-dataset de um único modelo.
- ``benchmark``: roda o benchmark cross-dataset de todos os modelos.
- ``evaluate`` : consolida as matrizes em um resumo in/out-of-domain.

Examples
--------
>>> python -m src.main prepare                    # doctest: +SKIP
>>> python -m src.main train --model tfidf_logreg # doctest: +SKIP
>>> python -m src.main benchmark                   # doctest: +SKIP
>>> python -m src.main evaluate                     # doctest: +SKIP
"""

from __future__ import annotations

import argparse
import logging
import os

from src.config.environment import seed_everything
from src.config.logging import configure_logging
from src.config.settings import load_settings
from src.evaluation.reports import build_benchmark_summary
from src.models.factory import list_models
from src.pipelines.benchmark import run_benchmark
from src.pipelines.preparation import (
    run_bronze_to_silver,
    run_preparation,
    run_silver_to_gold,
)

logger = logging.getLogger(__name__)

os.environ["TORCH_USE_CUDA_DSA"] = "1"  # evita OOM em GPUs com memória limitada (ex.: 4GB)


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos com os subcomandos do pipeline.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado com os subcomandos ``prepare``/``train``/
        ``benchmark``/``evaluate``.
    """
    parser = argparse.ArgumentParser(
        prog="sentibench",
        description="Benchmark de análise de sentimento em PT-BR (TF-IDF vs. BERTimbau).",
    )
    # Parent com --domains (após o subcomando, evitando ambiguidade do argparse).
    domains_parent = argparse.ArgumentParser(add_help=False)
    domains_parent.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Subconjunto de domínios a usar (padrão: todos do config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Nível de log (padrão: o definido em configs/logging.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare",
        parents=[domains_parent],
        help="Prepara os corpora (bronze -> silver -> gold).",
    )

    subparsers.add_parser(
        "bronze-to-silver",
        parents=[domains_parent],
        help="Limpa e valida o bronze, gerando o silver.",
    )
    subparsers.add_parser(
        "silver-to-gold",
        parents=[domains_parent],
        help="Amostra e divide o silver, gerando o gold.",
    )

    train_parser = subparsers.add_parser(
        "train",
        parents=[domains_parent],
        help="Roda o benchmark de um modelo.",
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=list(list_models()),
        help="Modelo a treinar/avaliar.",
    )
    train_parser.add_argument(
        "--no-track",
        action="store_true",
        help="Desativa o rastreamento com MLflow.",
    )

    bench_parser = subparsers.add_parser(
        "benchmark",
        parents=[domains_parent],
        help="Roda o benchmark completo.",
    )
    bench_parser.add_argument(
        "--no-track",
        action="store_true",
        help="Desativa o rastreamento com MLflow.",
    )

    subparsers.add_parser("evaluate", help="Consolida as matrizes em um resumo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argumentos da linha de comando; se ``None``, usa ``sys.argv``.

    Returns
    -------
    int
        Código de saída (0 em sucesso).
    """
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    settings = load_settings()
    seed_everything(settings.project.random_seed)

    if args.command == "prepare":
        run_preparation(domains=args.domains, settings=settings)
    elif args.command == "bronze-to-silver":
        run_bronze_to_silver(domains=args.domains, settings=settings)
    elif args.command == "silver-to-gold":
        run_silver_to_gold(domains=args.domains, settings=settings)
    elif args.command == "train":
        run_benchmark(
            [args.model],
            domains=args.domains,
            settings=settings,
            track=not args.no_track,
        )
    elif args.command == "benchmark":
        run_benchmark(
            list(list_models()),
            domains=args.domains,
            settings=settings,
            track=not args.no_track,
        )
    elif args.command == "evaluate":
        build_benchmark_summary(list(list_models()))

    logger.info("Comando '%s' finalizado com sucesso.", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
