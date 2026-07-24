# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Estrutura inicial do projeto (configs, `src/`, testes, CI, documentação).
- Baseline TF-IDF + Regressão Logística (`src/models/tfidf_logreg.py`).
- Fine-tuning do BERTimbau atrás de interface comum (`src/models/bertimbau.py`).
- Pipeline de preparação dos corpora (limpeza, validação, amostragem, split).
- Pipeline de benchmark cross-dataset (treina em N domínios, testa em N).
- Avaliação rigorosa: F1-macro com IC por bootstrap e teste de McNemar.
- Rastreamento de experimentos com MLflow.
- Figuras: matriz cross-dataset e matrizes de confusão.
- Contratos de dados (pandera) para os estágios bruto e processado.
- Suíte de testes (unitários, integração, property-based e comportamentais).

[Não lançado]: https://github.com/luanfreitas/sentimento-ptbr-benchmark/compare/HEAD
