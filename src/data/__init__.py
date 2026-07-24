"""Carregamento, amostragem e divisão dos corpora de sentimento.

Módulos
-------
loader
    Lê os CSVs brutos com polars, valida o contrato e limpa o texto.
sampler
    Amostragem estratificada por polaridade para viabilizar o benchmark.
splitter
    Divisão treino/teste estratificada e determinística por domínio.
"""

from src.data.loader import load_raw_domain, prepare_domain
from src.data.sampler import stratified_sample
from src.data.splitter import train_test_split_domain

__all__ = [
    "load_raw_domain",
    "prepare_domain",
    "stratified_sample",
    "train_test_split_domain",
]
