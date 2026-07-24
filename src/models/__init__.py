"""Modelos do benchmark sob uma interface comum.

Ambos os modelos implementam a interface ``SentimentClassifier``, o que permite
que o pipeline de benchmark os trate de forma intercambiável.

Módulos
-------
base
    Interface abstrata ``SentimentClassifier``.
tfidf_logreg
    Baseline clássico: TF-IDF + Regressão Logística (scikit-learn).
bertimbau
    Fine-tuning do BERTimbau (Hugging Face Transformers + PyTorch).
factory
    Cria uma instância de modelo a partir do nome e da configuração.
persistence
    Salva e carrega modelos treinados.
"""

from src.models.base import SentimentClassifier
from src.models.factory import build_model, list_models

__all__ = ["SentimentClassifier", "build_model", "list_models"]
