"""Testes da CLI (parser de argumentos)."""

from __future__ import annotations

import pytest

from src.main import build_parser


@pytest.mark.smoke
def test_parser_requires_command() -> None:
    """A CLI exige um subcomando."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_prepare() -> None:
    """O subcomando ``prepare`` é reconhecido."""
    args = build_parser().parse_args(["prepare"])
    assert args.command == "prepare"


def test_parser_train_requires_model() -> None:
    """O subcomando ``train`` exige ``--model``."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


def test_parser_train_with_valid_model() -> None:
    """``train --model tfidf_logreg`` é aceito."""
    args = build_parser().parse_args(["train", "--model", "tfidf_logreg"])
    assert args.model == "tfidf_logreg"
    assert args.command == "train"


def test_parser_rejects_invalid_model() -> None:
    """Um modelo fora das escolhas é rejeitado."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "inexistente"])


def test_parser_domains_option() -> None:
    """A opção ``--domains`` (após o subcomando) aceita múltiplos valores."""
    args = build_parser().parse_args(["prepare", "--domains", "b2w", "olist"])
    assert args.domains == ["b2w", "olist"]
