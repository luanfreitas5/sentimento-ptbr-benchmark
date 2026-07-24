"""Interface abstrata comum aos modelos de classificação de sentimento.

Definir um contrato único (fit/predict/predict_proba/save/load) desacopla o
pipeline de benchmark das implementações concretas (TF-IDF + LogReg e
BERTimbau), aplicando o princípio da inversão de dependência (SOLID).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np


class SentimentClassifier(ABC):
    """Contrato comum a todos os classificadores de sentimento do benchmark.

    Attributes
    ----------
    name : ClassVar[str]
        Identificador curto do modelo (ex.: ``tfidf_logreg``, ``bertimbau``).
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> SentimentClassifier:
        """Treina o modelo.

        Parameters
        ----------
        texts : Sequence[str]
            Textos de treino.
        labels : Sequence[int]
            Rótulos de polaridade (0/1) alinhados a ``texts``.

        Returns
        -------
        SentimentClassifier
            O próprio modelo treinado (permite encadeamento).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz os rótulos de polaridade para os textos.

        Parameters
        ----------
        texts : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Vetor de rótulos previstos (int) de shape ``(n,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz as probabilidades por classe.

        Parameters
        ----------
        texts : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Matriz de probabilidades de shape ``(n, 2)``, colunas na ordem
            ``[negativo, positivo]``.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persiste o modelo treinado no diretório indicado.

        Parameters
        ----------
        path : Path
            Diretório de destino (criado se não existir).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> SentimentClassifier:
        """Carrega um modelo previamente salvo.

        Parameters
        ----------
        path : Path
            Diretório de onde carregar o modelo.

        Returns
        -------
        SentimentClassifier
            Instância pronta para inferência.
        """
        raise NotImplementedError
