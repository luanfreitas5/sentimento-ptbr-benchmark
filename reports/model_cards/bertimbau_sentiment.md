# Model Card — BERTimbau para Sentimento PT-BR

> Modelo gerado por este projeto de benchmark. Preencha os valores marcados com
> `[preencher]` após a execução, a partir de `reports/metrics/`.

## Detalhes do modelo

- **Nome:** BERTimbau fine-tuned para classificação de polaridade (binária).
- **Base:** `neuralmind/bert-base-portuguese-cased` (BERTimbau base).
- **Tarefa:** classificação de sentimento (0 = negativo, 1 = positivo) em PT-BR.
- **Baseline comparado:** TF-IDF + Regressão Logística.
- **Versão / commit:** `[preencher: git SHA]`.
- **Hash dos dados de treino:** `[preencher: ver data/processed/manifest.json]`.

## Uso pretendido

- **Uso primário:** classificar a polaridade de avaliações/opiniões curtas em
  português brasileiro (e-commerce, apps, filmes).
- **Usuários:** pesquisadores e engenheiros de NLP construindo termômetros de
  sentimento sobre texto em PT-BR.

## Uso fora de escopo

- Textos que não sejam avaliações/opiniões (ex.: notícias, jurídico, saúde).
- Detecção de emoção fina, ironia/sarcasmo ou aspectos (ABSA).
- Decisões automatizadas com impacto sobre pessoas sem revisão humana.

## Dados de treino

- Corpora públicos de sentimento em PT-BR (ver o Datasheet correspondente).
- Amostragem estratificada por polaridade; split treino/teste estratificado.

## Avaliação

Métrica principal: **F1-macro** (classes desbalanceadas). Reportar sempre com
IC 95% (bootstrap).

### Desempenho por domínio (in-domain)

| Domínio | F1-macro | IC 95% |
|---|---|---|
| b2w | `[preencher]` | `[preencher]` |
| buscape | `[preencher]` | `[preencher]` |
| olist | `[preencher]` | `[preencher]` |
| utlc_apps | `[preencher]` | `[preencher]` |
| utlc_movies | `[preencher]` | `[preencher]` |

### Generalização cross-dataset

- F1 médio in-domain: `[preencher]`
- F1 médio out-of-domain: `[preencher]`
- **Gap de generalização:** `[preencher]`

### Significância vs. baseline (McNemar, in-domain)

- p-valor por domínio: `[preencher: reports/metrics/significance_mcnemar.json]`
- Conclusão: `[preencher: a diferença é / não é significativa a α=0,05]`

## Limitações conhecidas

- Sensível a mudança de domínio (ver gap de generalização).
- Reviews neutras foram removidas na construção do dataset — o modelo é binário.
- Pode refletir vieses presentes nos corpora de origem.

## Considerações éticas

- Não usar para decisões de alto impacto sem revisão humana.
- Auditar por subgrupos/domínios antes de qualquer implantação.
- Sem PII: os textos são opiniões públicas; não inferir identidade.
