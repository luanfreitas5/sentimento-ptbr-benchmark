# Datasheet — Brazilian Portuguese Sentiment Analysis Datasets

Baseado em *Datasheets for Datasets* (Gebru et al., 2018).

## Motivação

- **Propósito:** avaliar classificação de sentimento em português brasileiro em
  múltiplos domínios, permitindo estudo de generalização cross-dataset.
- **Fonte:** [Kaggle — ptbr-sentiment-analysis-datasets](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets),
  que reúne cinco corpora públicos de avaliações rotuladas.

## Composição

- **Domínios (5):** `b2w`, `buscape`, `olist`, `utlc_apps`, `utlc_movies`.
- **Instância:** uma avaliação textual de usuário com polaridade e nota.
- **Colunas relevantes:** `review_text` (texto), `polarity` (0/1), `rating` (1–5).
  As colunas originais incluem ainda texto pré-processado/tokenizado e índices de
  k-fold, não usadas por este projeto.
- **Rótulos:** polaridade binária; reviews neutras foram descartadas na
  construção original.
- **Desbalanceamento:** predominância da classe positiva — motiva o uso de
  F1-macro e `class_weight="balanced"` no baseline.

## Processo de coleta

- Avaliações públicas de usuários coletadas das respectivas plataformas pelos
  autores originais dos corpora. Este projeto **não** coleta dados novos.

## Pré-processamento (neste projeto)

- Remoção de nulos e de textos muito curtos (`min_chars`).
- Limpeza leve para o TF-IDF (minúsculas, URLs/menções, repetições; acentos
  preservados). O BERTimbau usa o texto original.
- Amostragem estratificada por polaridade e split treino/teste estratificado.
- Deduplicação por texto limpo.

## Distribuição e licenciamento

- Consulte a licença de cada corpus na página do Kaggle antes de redistribuir.
- Uso neste projeto: pesquisa/portfólio, sem redistribuição dos dados brutos.

## Privacidade / LGPD

- Os textos são opiniões públicas de produtos/serviços. **Não** contêm, por
  design, identificadores diretos.
- O projeto **não** registra PII em logs, figuras ou saídas.
- Quasi-identificadores devem ser evitados em qualquer análise derivada.

## Manutenção

- Versão dos dados rastreada pelo hash SHA-256 em `data/processed/manifest.json`.
