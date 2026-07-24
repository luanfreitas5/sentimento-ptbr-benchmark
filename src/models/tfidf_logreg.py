"""Baseline clássico: TF-IDF + Regressão Logística (scikit-learn).

É o benchmark simples que qualquer modelo mais complexo precisa superar para
justificar seu custo. Encapsula um ``Pipeline`` do scikit-learn atrás da
interface ``SentimentClassifier``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.settings import TfidfLogRegConfig
from src.constants.labels import LABEL_ORDER
from src.exceptions.model import ModelNotFittedError
from src.models.base import SentimentClassifier

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "tfidf_logreg.joblib"


class TfidfLogRegClassifier(SentimentClassifier):
    """Classificador TF-IDF + Regressão Logística.

    Parameters
    ----------
    config : TfidfLogRegConfig
        Hiperparâmetros validados do vetorizador e do estimador.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Examples
    --------
    >>> from src.config.settings import TfidfLogRegConfig
    >>> clf = TfidfLogRegClassifier(TfidfLogRegConfig())
    >>> _ = clf.fit(["ótimo", "péssimo", "adorei", "horrível"], [1, 0, 1, 0])
    >>> int(clf.predict(["maravilhoso"])[0]) in (0, 1)
    True
    """

    name: ClassVar[str] = "tfidf_logreg"

    def __init__(self, config: TfidfLogRegConfig, *, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Constrói o ``Pipeline`` sklearn a partir da configuração."""
        tfidf_cfg = self.config.tfidf
        logreg_cfg = self.config.logreg
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tfidf_cfg.ngram_range,
                        min_df=tfidf_cfg.min_df,
                        max_df=tfidf_cfg.max_df,
                        max_features=tfidf_cfg.max_features,
                        sublinear_tf=tfidf_cfg.sublinear_tf,
                        strip_accents=tfidf_cfg.strip_accents,
                    ),
                ),
                (
                    "logreg",
                    LogisticRegression(
                        C=logreg_cfg.C,
                        max_iter=logreg_cfg.max_iter,
                        class_weight=logreg_cfg.class_weight,
                        solver=logreg_cfg.solver,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> TfidfLogRegClassifier:
        """Treina o pipeline TF-IDF + LogReg.

        Parameters
        ----------
        texts : Sequence[str]
            Textos de treino (recomenda-se o texto limpo).
        labels : Sequence[int]
            Rótulos de polaridade (0/1).

        Returns
        -------
        TfidfLogRegClassifier
            O próprio modelo treinado.
        """
        logger.info("Treinando baseline TF-IDF + LogReg em %d exemplos", len(texts))
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))
        return self

    def _check_fitted(self) -> Pipeline:
        """Garante que o modelo foi treinado antes de inferir/salvar."""
        if self.pipeline is None:
            raise ModelNotFittedError("O modelo TF-IDF + LogReg ainda não foi treinado.")
        return self.pipeline

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz os rótulos de polaridade. Ver ``SentimentClassifier.predict``."""
        return np.asarray(self._check_fitted().predict(list(texts)))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz probabilidades por classe na ordem ``[negativo, positivo]``."""
        pipeline = self._check_fitted()
        proba = pipeline.predict_proba(list(texts))
        # Reordena as colunas para a ordem canônica de LABEL_ORDER.
        classes = list(pipeline.classes_)
        order = [classes.index(label) for label in LABEL_ORDER]
        return np.asarray(proba)[:, order]

    def save(self, path: Path) -> None:
        """Serializa o pipeline com joblib em ``path/tfidf_logreg.joblib``."""
        pipeline = self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": pipeline, "config": self.config, "seed": self.seed}, path / _MODEL_FILENAME
        )
        logger.info("Baseline salvo em %s", path / _MODEL_FILENAME)

    @classmethod
    def load(cls, path: Path) -> TfidfLogRegClassifier:
        """Carrega um baseline previamente salvo com ``save``."""
        path = Path(path)
        artifact = joblib.load(path / _MODEL_FILENAME)
        model = cls(artifact["config"], seed=artifact["seed"])
        model.pipeline = artifact["pipeline"]
        return model
