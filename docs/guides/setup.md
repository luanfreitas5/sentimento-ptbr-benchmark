# Setup

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) para gestão de ambiente e dependências
- (Opcional) GPU NVIDIA para o fine-tuning do BERTimbau

## Instalação

```bash
# Baseline (TF-IDF + LogReg), sem torch/transformers:
make install

# Baseline + BERTimbau (torch/transformers/accelerate/datasets):
make install-bert

# Tudo (inclui o extra dvc):
make install-all
```

Instale os hooks de qualidade antes de commitar:

```bash
make hooks
uv run detect-secrets scan > .secrets.baseline
```

## Dados

Baixe os cinco CSVs do dataset
[Brazilian Portuguese Sentiment Analysis Datasets](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets)
e coloque-os em `data/raw/`:

```
data/raw/
├── b2w.csv
├── buscape.csv
├── olist.csv
├── utlc_apps.csv
└── utlc_movies.csv
```

Os dados **não** são versionados no Git. O manifest gerado na preparação
(`data/processed/manifest.json`) guarda o hash SHA-256 de cada corpus para
rastrear a versão.

## Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste conforme necessário. Nenhuma chave é
obrigatória para o baseline; o BERTimbau usa um modelo público.
