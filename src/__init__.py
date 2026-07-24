"""Pacote-fonte do benchmark de análise de sentimento em PT-BR.

Compara um baseline TF-IDF + Regressão Logística contra o fine-tuning do
BERTimbau, com avaliação cross-dataset (treina em um domínio, testa em outro)
como diferencial experimental.

Subpacotes principais
----------------------
config
    Carregamento e validação de configurações, logging, caminhos e sementes.
constants
    Nomes de colunas, rótulos e identificadores de datasets.
data
    Carregamento, amostragem e divisão dos corpora.
preprocessing
    Limpeza e normalização de texto.
schemas
    Contratos de dados (pandera) por estágio do pipeline.
models
    Baseline TF-IDF + LogReg e transformer BERTimbau sob uma interface comum.
evaluation
    Métricas, incerteza (bootstrap) e testes de significância (McNemar).
experiment
    Rastreamento de experimentos com MLflow.
visualization
    Figuras do benchmark (matriz cross-dataset, matrizes de confusão).
pipelines
    Orquestração ponta a ponta (preparação, treino, benchmark, avaliação).
"""

__version__ = "0.1.0"
