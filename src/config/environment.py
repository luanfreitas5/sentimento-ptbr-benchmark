"""Reprodutibilidade determinística: fixa todas as fontes de aleatoriedade.

``random_state`` sozinho não é reprodutibilidade. Este módulo semeia
``random``, ``numpy``, ``PYTHONHASHSEED`` e, se presentes, ``torch`` e
``transformers``.

Examples
--------
>>> from src.config.environment import seed_everything
>>> seed_everything(42)
"""

from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np

RANDOM_SEED: int = 42


def seed_everything(seed: int = RANDOM_SEED, *, deterministic_torch: bool = False) -> None:
    """Fixa todas as sementes para garantir reprodutibilidade.

    Parameters
    ----------
    seed : int, optional
        Semente global, by default ``42``.
    deterministic_torch : bool, optional
        Se ``True``, força algoritmos determinísticos no PyTorch (mais lento,
        porém reprodutível). Ignorado se o torch não estiver instalado.

    Notes
    -----
    Também define ``PYTHONHASHSEED`` para tornar o hashing determinístico.

    Examples
    --------
    >>> seed_everything(123)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed=123)
    rng.normal()

    with suppress(ImportError):  # opcional: só se o extra `bert` (transformers) estiver instalado.
        from transformers import set_seed  # pyright: ignore[reportMissingImports]

        set_seed(seed)

    with suppress(ImportError):  # opcional: só se o extra `bert` (torch) estiver instalado.
        import torch  # pyright: ignore[reportMissingImports]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
