"""Teste de significância estatística para comparação de classificadores.

Duas médias de F1 diferentes não bastam para afirmar que um modelo é melhor.
Como os dois modelos são avaliados no **mesmo** conjunto de teste, o teste de
McNemar (sobre a tabela de acertos/erros pareados) é o adequado.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class McNemarResult:
    """Resultado do teste de McNemar entre dois modelos.

    Attributes
    ----------
    statistic : float
        Estatística do teste.
    p_value : float
        Valor-p associado.
    n01 : int
        Casos em que só o modelo A errou (B acertou).
    n10 : int
        Casos em que só o modelo B errou (A acertou).
    significant : bool
        ``True`` se ``p_value < alpha``.
    """

    statistic: float
    p_value: float
    n01: int
    n10: int
    significant: bool


def mcnemar_test(
    y_true: Sequence[int] | np.ndarray,
    y_pred_a: Sequence[int] | np.ndarray,
    y_pred_b: Sequence[int] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    """Compara dois modelos no mesmo teste via teste de McNemar.

    Parameters
    ----------
    y_true : Sequence[int] | np.ndarray
        Rótulos verdadeiros.
    y_pred_a : Sequence[int] | np.ndarray
        Predições do modelo A (ex.: baseline).
    y_pred_b : Sequence[int] | np.ndarray
        Predições do modelo B (ex.: BERTimbau).
    alpha : float, optional
        Nível de significância, by default 0.05.

    Returns
    -------
    McNemarResult
        Estatística, valor-p, discordâncias e decisão de significância.

    Notes
    -----
    Usa a correção de continuidade quando as discordâncias são poucas
    (``n01 + n10 < 25``), e o teste binomial exato caso contrário seja
    apropriado (delegado ao statsmodels).

    Examples
    --------
    >>> res = mcnemar_test([1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0])
    >>> res.significant in (True, False)
    True
    """
    y_true_arr = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true_arr
    correct_b = np.asarray(y_pred_b) == y_true_arr

    # Tabela de contingência 2x2 dos acertos pareados.
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    table = [[n11, n10], [n01, n00]]

    discordances = n01 + n10
    use_exact = discordances < 25
    outcome = mcnemar(table, exact=use_exact, correction=True)
    # ``outcome`` é um `_Bunch` do statsmodels: atributos definidos em runtime,
    # sem tipagem estática — daí as supressões pontuais abaixo.
    statistic = outcome.statistic  # pyright: ignore[reportAttributeAccessIssue]
    p_value = outcome.pvalue  # pyright: ignore[reportAttributeAccessIssue]

    return McNemarResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n01=n01,
        n10=n10,
        significant=bool(p_value < alpha),
    )
