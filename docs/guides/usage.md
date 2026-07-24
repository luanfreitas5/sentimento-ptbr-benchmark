# Uso

O pipeline é exposto por quatro subcomandos da CLI (`src/main.py`), com atalhos
equivalentes no `Makefile`.

## 1. Preparação dos corpora

Limpa o texto, valida os contratos, amostra de forma estratificada, divide em
treino/teste e salva em parquet (`data/processed/<domínio>/{train,test}.parquet`).

```bash
make prepare
# ou: uv run python -m src.main prepare --domains b2w olist
```

## 2. Treino/avaliação de um modelo

Roda o benchmark cross-dataset de um único modelo.

```bash
make baseline                                   # TF-IDF + LogReg
make bert                                        # BERTimbau (requer make install-bert)
# ou: uv run python -m src.main train --model tfidf_logreg
```

## 3. Benchmark completo

Roda todos os modelos e compara-os in-domain via teste de McNemar.

```bash
make benchmark
# ou: uv run python -m src.main benchmark --no-track   # desativa o MLflow
```

## 4. Consolidação

Gera o resumo in-domain vs. out-of-domain e o *gap* de generalização.

```bash
make evaluate
```

## Resultados

- `reports/figures/cross_dataset_<modelo>.png` — heatmap da matriz cross-dataset;
- `reports/metrics/benchmark_<modelo>.json` — métricas por célula com IC 95%;
- `reports/metrics/significance_mcnemar.json` — significância baseline vs. BERTimbau;
- `reports/metrics/summary.json` — resumo in/out-domain;
- `make mlflow-ui` — inspeciona parâmetros/métricas de cada execução.

## Configuração

Ajuste `configs/config.yaml` (domínios, amostragem, split) e
`configs/model_params.yaml` (hiperparâmetros). Todos os valores são validados
por Pydantic no *startup* — uma configuração inválida falha cedo, com erro claro.
