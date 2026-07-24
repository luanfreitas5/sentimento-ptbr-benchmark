# Arquitetura e decisões de projeto

## Princípios

- **Interface comum aos modelos.** `SentimentClassifier` define
  `fit`/`predict`/`predict_proba`/`save`/`load`. Baseline e BERTimbau a
  implementam, e o pipeline de benchmark os trata de forma intercambiável
  (inversão de dependência — SOLID). Um novo modelo entra registrando uma
  linha na `factory`.
- **Validar dados, não só código.** Contratos `pandera` (`src/schemas/`) checam
  tipos, faixas e categorias na entrada (pós-carga) e na saída (dataset
  processado), falhando cedo em vez de propagar corrupção.
- **Configuração validada.** YAMLs em `configs/` são carregados e validados por
  Pydantic no *startup*; um valor inválido falha imediatamente.
- **Reprodutibilidade determinística.** *Seed everything*, lock file commitado e
  hash SHA-256 dos corpora no manifest da preparação.

## Fluxo de dados

```
data/raw/*.csv
   │  loader (limpeza + contrato bruto)
   ▼
sampler (estratificado)  ─►  contrato processado  ─►  splitter
   │
   ▼
data/processed/<domínio>/{train,test}.parquet
   │  benchmark: treina em cada domínio, testa em todos
   ▼
reports/metrics/*.json  +  reports/figures/*.png  +  mlruns/
```

## Escolhas técnicas defensáveis

- **Métrica principal: F1-macro.** Os corpora pendem fortemente para a classe
  positiva; F1-macro trata ambas as classes igualmente.
- **Texto por modelo.** O baseline TF-IDF usa texto **limpo** (minúsculas,
  remoção de URLs/ruído, colapso de repetições, acentos preservados); o
  BERTimbau usa o texto **original**, pois seu tokenizer WordPiece foi treinado
  sobre texto cru.
- **Incerteza e significância.** Toda métrica vem com IC por bootstrap; a
  comparação entre modelos usa McNemar (mesmo conjunto de teste, predições
  pareadas) — não se afirma "melhor" a partir de um único número.
- **Amostragem.** Os corpora somam milhões de linhas; a amostragem estratificada
  por polaridade viabiliza o benchmark (sobretudo o BERTimbau) preservando a
  proporção de classes. `sample_size: null` usa o corpus inteiro.
- **Dependência opcional.** `torch`/`transformers` ficam no extra `bert`; o
  baseline roda sem eles. Os imports do BERTimbau são preguiçosos.

## Estrutura de pacotes

Ver o `README.md` para a árvore de `src/`. Cada subpacote tem `__init__.py` com
docstring descrevendo seu propósito e módulos.
