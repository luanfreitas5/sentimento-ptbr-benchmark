"""Testes de amostragem e divisão dos corpora."""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.data.sampler import stratified_sample
from src.data.splitter import train_test_split_domain


def _balanced_df(n_per_class: int = 50) -> pl.DataFrame:
    """DataFrame balanceado com colunas ``polarity`` e ``text_clean``."""
    return pl.DataFrame(
        {
            "polarity": [0] * n_per_class + [1] * n_per_class,
            "text_clean": [f"neg {i}" for i in range(n_per_class)]
            + [f"pos {i}" for i in range(n_per_class)],
        }
    )


@pytest.mark.smoke
def test_stratified_sample_respects_size() -> None:
    """A amostra tem aproximadamente o tamanho pedido."""
    df = _balanced_df(50)
    sampled = stratified_sample(df, 20, seed=42)
    assert 16 <= sampled.height <= 24


def test_stratified_sample_none_returns_all() -> None:
    """``sample_size=None`` retorna todas as linhas (apenas embaralhadas)."""
    df = _balanced_df(30)
    assert stratified_sample(df, None, seed=1).height == df.height


def test_stratified_sample_preserves_both_classes() -> None:
    """Ambas as classes permanecem representadas após a amostragem."""
    df = _balanced_df(50)
    sampled = stratified_sample(df, 20, seed=42)
    assert sampled["polarity"].n_unique() == 2


@pytest.mark.smoke
def test_train_test_split_sizes() -> None:
    """Os tamanhos dos splits respeitam ``test_size``."""
    df = _balanced_df(50)
    train, test = train_test_split_domain(df, test_size=0.2, seed=42)
    assert train.height == 80
    assert test.height == 20


def test_train_test_split_is_deterministic() -> None:
    """A mesma semente produz a mesma divisão."""
    df = _balanced_df(40)
    train_a, test_a = train_test_split_domain(df, seed=7)
    train_b, test_b = train_test_split_domain(df, seed=7)
    assert train_a["text_clean"].to_list() == train_b["text_clean"].to_list()
    assert test_a["text_clean"].to_list() == test_b["text_clean"].to_list()


def test_train_test_split_raises_on_tiny_input() -> None:
    """Um DataFrame com uma linha não pode ser dividido."""
    df = pl.DataFrame({"polarity": [1], "text_clean": ["x"]})
    with pytest.raises(ValueError, match="insuficiente"):
        train_test_split_domain(df)


@given(size=st.floats(min_value=0.1, max_value=0.5))
def test_split_partitions_without_overlap(size: float) -> None:
    """Invariante: treino e teste particionam o conjunto sem sobreposição."""
    df = _balanced_df(50)
    train, test = train_test_split_domain(df, test_size=size, seed=3)
    assert train.height + test.height == df.height
    overlap = set(train["text_clean"].to_list()) & set(test["text_clean"].to_list())
    assert not overlap
