# 🇧🇷 Benchmark de Análise de Sentimento em PT-BR

> TF-IDF + Regressão Logística **vs.** fine-tuning do **BERTimbau**, com avaliação **cross-dataset** (treina em um domínio, testa em outro) como diferencial experimental.

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](.github/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](.github/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](pyproject.toml)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 Motivação

Análise de sentimento em **português** é um recurso ainda escasso frente ao inglês. Este projeto entrega um **benchmark reprodutível e publicável** que responde a duas perguntas de valor prático:

1. **Vale a pena o custo do BERTimbau?** — quanto um transformer fine-tuned supera um baseline clássico (TF-IDF + Regressão Logística), e a diferença é **estatisticamente significativa**?
2. **Os modelos generalizam entre domínios?** — treinar em avaliações de e-commerce e aplicar em avaliações de filmes funciona? O **diferencial** aqui é a **matriz cross-dataset**: treina em cada um dos 5 corpora e testa em **todos** eles.

## 🧪 O diferencial: avaliação cross-dataset

Cada modelo é treinado em um domínio e avaliado em **todos** os domínios, produzindo uma matriz `N × N` de F1-macro:

- **Diagonal** → desempenho *in-domain* (mesmo domínio de treino e teste);
- **Fora da diagonal** → desempenho *out-of-domain* (transferência entre domínios);
- **Gap de generalização** = média in-domain − média out-of-domain.

Isso expõe algo que um único split esconde: um modelo pode ir muito bem no seu domínio e desabar fora dele.

## 📊 Dados

[Brazilian Portuguese Sentiment Analysis Datasets](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets) — cinco corpora rotulados em polaridade (0 = negativo, 1 = positivo) e nota (1–5):

| Domínio | Fonte | Conteúdo |
|---|---|---|
| `b2w` | B2W / Americanas | e-commerce (varejo) |
| `buscape` | Buscapé | comparador de preços |
| `olist` | Olist | marketplace |
| `utlc_apps` | UTLC | avaliações de apps |
| `utlc_movies` | UTLC | avaliações de filmes |

> Baixe os CSVs do Kaggle para `data/raw/`. Os dados **não** são versionados no Git (ver `.gitignore`); em produção use DVC.

## 🏗️ Arquitetura

```
src/
├── config/         # settings (Pydantic), paths, logging (rich), seeds
├── constants/      # colunas, rótulos, catálogo de domínios
├── schemas/        # contratos de dados (pandera): raw e processed
├── preprocessing/  # limpeza de texto PT-BR (para o TF-IDF)
├── data/           # loader, amostragem estratificada, split
├── models/         # interface comum + TF-IDF/LogReg + BERTimbau + factory
├── metrics/        # F1-macro e IC por bootstrap
├── evaluation/     # avaliador, McNemar, consolidação in/out-domain
├── experiment/     # rastreamento MLflow
├── visualization/  # matriz cross-dataset, matrizes de confusão
├── pipelines/      # preparação + benchmark ponta a ponta
└── main.py         # CLI (prepare / train / benchmark / evaluate)
```

Os dois modelos implementam a mesma interface `SentimentClassifier` (`fit`/`predict`/`predict_proba`/`save`/`load`), então o pipeline os trata de forma intercambiável (inversão de dependência).

## 🚀 Quickstart

```bash
# 1. Ambiente (uv). Baseline sem torch:
make install
# ...ou incluindo o fine-tuning do BERTimbau (torch/transformers):
make install-bert

# 2. Baixe os CSVs do Kaggle para data/raw/ e prepare os corpora:
make prepare

# 3. Baseline (TF-IDF + LogReg) em todos os domínios:
make baseline

# 4. BERTimbau (requer make install-bert e, idealmente, GPU):
make bert

# 5. Benchmark cross-dataset completo + consolidação:
make benchmark
make evaluate

# 6. Inspecione os experimentos no MLflow:
make mlflow-ui
```

Também disponível como CLI:

```bash
uv run python -m src.main prepare
uv run python -m src.main train --model tfidf_logreg
uv run python -m src.main benchmark --no-track
uv run python -m src.main evaluate
```

## 📈 Saídas

- `reports/figures/cross_dataset_<modelo>.png` — heatmap da matriz cross-dataset;
- `reports/metrics/benchmark_<modelo>.json` — matriz de F1 + métricas por célula (com IC 95%);
- `reports/metrics/significance_mcnemar.json` — teste de McNemar (baseline vs. BERTimbau, in-domain);
- `reports/metrics/summary.json` — F1 médio in-domain vs. out-of-domain e gap de generalização;
- `mlruns/` — parâmetros, métricas e artefatos por execução (MLflow).

## 🔬 Rigor experimental (senior bar)

- **Incerteza:** toda métrica reportada com intervalo de confiança (bootstrap), nunca um ponto solto;
- **Significância:** McNemar para decidir se o BERTimbau realmente supera o baseline;
- **Slices:** desempenho reportado por domínio, não só agregado;
- **Reprodutibilidade:** *seed everything*, lock file commitado, hash SHA-256 dos corpora no manifest;
- **Contratos de dados:** `pandera` valida a entrada e a saída de cada etapa;
- **Métrica principal:** F1-macro — os corpora pendem para a classe positiva, e ambas as classes importam.

## 🛠️ Desenvolvimento

```bash
make quality   # ruff, basedpyright, bandit, vulture, xenon, interrogate
make test      # pytest com cobertura (≥ 80%)
make docs      # MkDocs Material (referência da API a partir das docstrings)
```

## 📚 Documentação

Guias e referência completa da API em **`docs/`** (publicados via GitHub Pages). Comece por [`docs/index.md`](docs/index.md).

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
