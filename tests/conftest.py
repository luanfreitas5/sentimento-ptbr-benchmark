"""Fixtures compartilhadas dos testes.

Fornecem DataFrames sintéticos pequenos e um modelo baseline já treinado, sem
depender de dados de produção nem de dependências pesadas (torch/transformers).
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config.settings import LogRegParams, TfidfLogRegConfig, TfidfParams
from src.models.tfidf_logreg import TfidfLogRegClassifier


def make_test_config() -> TfidfLogRegConfig:
    """Config do baseline adequada a datasets minúsculos (``min_df=1``)."""
    return TfidfLogRegConfig(
        tfidf=TfidfParams(ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=None),
        logreg=LogRegParams(),
    )


# Frases curtas com polaridade clara para testes determinísticos.
_POSITIVE = [
    "produto excelente adorei recomendo",
    "ótima qualidade entrega rápida muito bom",
    "maravilhoso superou expectativas perfeito",
    "amei chegou antes do prazo top",
    "excelente custo benefício compraria de novo",
]
_NEGATIVE = [
    "péssimo produto quebrado detestei",
    "horrível não recomendo dinheiro jogado fora",
    "muito ruim veio com defeito decepcionante",
    "atendimento terrível produto de baixa qualidade",
    "não funciona lixo arrependido da compra",
]


@pytest.fixture
def synthetic_texts() -> tuple[list[str], list[int]]:
    """Retorna textos sintéticos e rótulos balanceados (0/1)."""
    texts = _POSITIVE + _NEGATIVE
    labels = [1] * len(_POSITIVE) + [0] * len(_NEGATIVE)
    return texts, labels


@pytest.fixture
def raw_domain_df() -> pl.DataFrame:
    """DataFrame bruto sintético no esquema dos corpora."""
    return pl.DataFrame(
        {
            "review_text": ["Ótimo produto!", "Péssimo, não gostei", None, "Bom demais"],
            "polarity": [1.0, 0.0, 1.0, 1.0],
            "rating": [5.0, 1.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def trained_baseline(
    synthetic_texts: tuple[list[str], list[int]],
) -> TfidfLogRegClassifier:
    """Baseline TF-IDF + LogReg já treinado nos textos sintéticos."""
    texts, labels = synthetic_texts
    model = TfidfLogRegClassifier(make_test_config(), seed=42)
    return model.fit(texts, labels)
