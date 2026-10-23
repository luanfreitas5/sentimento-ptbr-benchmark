# Model Card — BERTimbau para Sentimento PT-BR v0.2.0

**Status:** DRAFT
**Bloqueios pendentes:** teste de significância formal (McNemar) e calibração — bloqueiam qualquer afirmação de "melhor que o baseline" em publicação; auditoria de fairness e rastreabilidade (Git SHA / run MLflow) — bloqueiam uso fora de pesquisa/portfólio.

## 1. Detalhes do modelo
- **Responsável / contato:** Luan Freitas — luan.mgf@gmail.com
- **Data:** 2026-10-23
- **Algoritmo:** `neuralmind/bert-base-portuguese-cased` (BERTimbau base) com fine-tuning para classificação binária; 2 épocas, batch 32, lr 2e-5, `max_length` 128 (`configs/model_params.yaml`). Um modelo por domínio de treino (5 modelos).
- **Rastreabilidade:** Git SHA `TODO(owner): registrar o commit que gerou reports/metrics/` · MLflow run `TODO(owner): ids em mlruns/mlflow.db` · hash dos dados: SHA-256 dos CSVs bronze em `data/bronze/manifest.json` (replicado em `data/gold/manifest.json`)
- **Licença:** código MIT; pesos do BERTimbau conforme o modelo base (ver Hugging Face)

## 2. Uso pretendido
- **Uso primário:** classificar a polaridade (negativo/positivo) de avaliações curtas de consumidores em português brasileiro (e-commerce, apps, filmes), como referência de benchmark.
- **Usuários pretendidos:** pesquisadores e engenheiros de NLP que constroem termômetros de sentimento em PT-BR.
- **Humano no circuito:** sim — as saídas apoiam análise agregada; não devem acionar decisões individuais automáticas.

## 3. Usos fora do escopo
- Textos que não sejam avaliações/opiniões (notícias, jurídico, saúde, redes sociais gerais).
- Detecção de emoção fina, ironia/sarcasmo, aspectos (ABSA) ou polaridade neutra — o modelo é binário.
- Decisões automatizadas com impacto sobre pessoas (moderação, crédito, atendimento) sem revisão humana.
- Aplicação a um domínio novo sem revalidação: treinar em um domínio e usar em outro perde até ~0,35 de F1-macro (ver §8).

## 4. Fatores
- **Grupos relevantes:** domínio de origem (`b2w`, `buscape`, `olist`, `utlc_apps`, `utlc_movies`). Os dados não trazem atributos demográficos dos autores.
- **Ambientes:** reviews de varejo/marketplace, comparador de preços, apps e filmes; texto curto, ortografia informal.

## 5. Métricas
- **Métrica principal:** F1-macro — os corpora pendem para a classe positiva (67–90% no teste) e errar a classe negativa é o erro mais caro para monitoramento de reputação; a acurácia mascararia esse erro.
- **Limiar de decisão:** TODO(owner): confirmar em `src/models/bertimbau.py` (presumivelmente argmax das probabilidades, ou seja, 0,5); nenhum limiar foi ajustado em validação.
- **Calibração:** não avaliada — TODO(owner): Brier e curva de confiabilidade em `reports/figures/`.
- **Incerteza:** IC 95% por bootstrap (1.000 reamostragens) no teste de cada domínio.

## 6. Dados de avaliação
Conjunto de teste de cada domínio: 4.000 reviews (split 80/20 estratificado, seed 42, a partir de uma amostra estratificada de 20.000 por domínio). Prevalência de positivos: b2w 69,7% · buscape 90,1% · olist 66,6% · utlc_apps 71,7% · utlc_movies 88,3%. Split aleatório, sem separação temporal. Avaliação **interna apenas; não há validação externa**.

## 7. Dados de treino
Cerca de 16.000 reviews por domínio, da mesma amostra descrita acima. Ver o [Datasheet](../datasheets/ptbr_sentiment_datasets.md).

## 8. Análise quantitativa
Fonte: `reports/metrics/benchmark_bertimbau.json` e `benchmark_tfidf_logreg.json`. Recorte = domínio de treino e teste iguais (in-domain), n = 4.000 cada.

| Recorte | n | F1-macro ± IC 95% (BERTimbau) | Baseline TF-IDF + LogReg |
|---|---|---|---|
| Média in-domain | 20.000 | 0,890 (IC da média: TODO) | 0,843 |
| b2w | 4.000 | 0,955 (0,947–0,962) | 0,914 (0,905–0,924) |
| buscape | 4.000 | 0,845 (0,823–0,865) | 0,787 (0,767–0,807) |
| olist | 4.000 | 0,938 (0,930–0,946) | 0,906 (0,896–0,915) |
| utlc_apps | 4.000 | 0,896 (0,886–0,906) | 0,861 (0,850–0,872) |
| utlc_movies | 4.000 | 0,817 (0,796–0,836) | 0,747 (0,727–0,765) |

**Generalização cross-dataset** (`summary.json`): F1 médio out-of-domain 0,796 (BERTimbau) vs. 0,722 (baseline); gap de generalização 0,095 vs. 0,121. Pior transferência: `utlc_movies → olist` = 0,600; melhor: `olist → b2w` = 0,938.

**Significância vs. baseline:** os ICs 95% não se sobrepõem em nenhum domínio, mas isso não substitui o McNemar — TODO(owner): gerar `reports/metrics/significance_mcnemar.json`.

## 9. Considerações éticas e fairness
- Atributos sensíveis auditados: nenhum — os corpora não trazem atributos dos autores; TODO(owner): auditar viés de dialeto/região e por tema (ex.: menções a grupos) antes de qualquer uso aplicado.
- Disparidades encontradas: apenas por domínio (§8); não há métrica de paridade por subgrupo de pessoas.
- Mitigação aplicada: nenhuma.
- Base legal LGPD: textos de opinião públicos, sem identificadores diretos por design; não registrar texto de usuários em logs (ver Datasheet §8).

## 10. Limitações, riscos e monitoramento
- **Limitações conhecidas:** forte queda fora do domínio, sobretudo envolvendo filmes; modelo binário (reviews neutras foram removidas na origem); uma seed e um split por domínio (variância entre treinos não medida); amostra de 20.000 por domínio, não o corpus completo; viés das plataformas de origem.
- **Mudanças de distribuição previstas:** vocabulário e gírias mudam com o tempo; o período de coleta dos corpora não é documentado, portanto o desvio temporal é desconhecido.
- **Monitoramento:** não há implantação; em caso de uso real, acompanhar F1-macro com amostra rotulada periódica e drift de comprimento/vocabulário (`evidently`).
- **Critério de retreino / descontinuação:** TODO(owner): definir limiar de queda de F1-macro; descontinuar se a validação externa ficar abaixo do baseline.
