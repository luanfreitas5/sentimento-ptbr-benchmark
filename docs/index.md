# Benchmark de Sentimento PT-BR

Comparação reprodutível entre um **baseline TF-IDF + Regressão Logística** e o
**fine-tuning do BERTimbau** para classificação de polaridade em português, com
**avaliação cross-dataset** como diferencial.

## Perguntas que o projeto responde

1. O BERTimbau supera o baseline clássico de forma **estatisticamente
   significativa** (teste de McNemar)?
2. O quanto cada modelo **generaliza entre domínios** (e-commerce, apps,
   filmes...)? A matriz cross-dataset e o *gap* de generalização quantificam
   isso.

## Navegação

- [Setup](guides/setup.md) — instalação do ambiente e download dos dados.
- [Uso](guides/usage.md) — executando o pipeline (prepare → benchmark → evaluate).
- [Arquitetura](guides/architecture.md) — decisões de projeto e organização do código.
- [Referência da API](reference.md) — documentação gerada a partir das docstrings.

## Diferencial: matriz cross-dataset

Cada modelo é treinado em um domínio e avaliado em **todos** os domínios,
produzindo uma matriz `N × N` de F1-macro. A diagonal mede o desempenho
*in-domain*; o restante mede a transferência *out-of-domain*.
