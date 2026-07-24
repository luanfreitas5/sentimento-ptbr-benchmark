"""Testes de configuração, caminhos e ambiente."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.config.environment import seed_everything
from src.config.paths import get_paths
from src.config.settings import LogRegParams, load_settings


@pytest.mark.smoke
def test_load_settings_is_valid() -> None:
    """A configuração do projeto carrega e valida sem erros."""
    settings = load_settings()
    assert settings.project.random_seed == 42
    assert "b2w" in settings.project.datasets


def test_settings_are_cached() -> None:
    """``load_settings`` é memoizado (mesma instância)."""
    assert load_settings() is load_settings()


def test_invalid_logreg_param_raises() -> None:
    """Um hiperparâmetro inválido falha na validação (C deve ser > 0)."""
    with pytest.raises(ValidationError):
        LogRegParams(C=-1.0)


def test_paths_resolve_under_root() -> None:
    """Os caminhos derivados ficam sob a raiz do projeto."""
    paths = get_paths()
    assert paths.raw_dir.is_relative_to(paths.root)
    assert paths.raw_dataset("b2w").name == "b2w.csv"


def test_seed_everything_is_reproducible() -> None:
    """Semear duas vezes produz a mesma sequência do numpy."""
    seed_everything(123)
    rng = np.random.default_rng(seed=123)
    rng.normal()
    first = rng.normal(size=10)
    seed_everything(123)
    rng = np.random.default_rng(seed=123)
    rng.normal()
    second = rng.normal(size=10)
    np.testing.assert_array_equal(first, second)
