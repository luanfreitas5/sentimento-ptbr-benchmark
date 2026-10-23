# 🗄️ Dados — arquitetura medalhão

Os dados **não** são versionados no Git (ver `.gitignore`); apenas os `.gitkeep`
e este README. Em produção, versione com DVC. Regra de ouro: **nunca modificar
o bronze** — sempre ler, transformar e gravar na camada seguinte.

```
data/
├── bronze/                 # camada 1 — como recebido (imutável)
│   ├── <dominio>.csv       #   b2w, buscape, olist, utlc_apps, utlc_movies
│   └── manifest.json       #   SHA-256 de cada CSV
├── silver/                 # camada 2 — limpo, tipado, deduplicado e validado (pandera)
│   ├── <dominio>.parquet
│   └── manifest.json       #   linhas, SHA-256 do parquet e do CSV bronze de origem
└── gold/                   # camada 3 — pronto para modelagem
    ├── <dominio>/{train,test}.parquet   # amostra estratificada + split estratificado
    └── manifest.json       #   tamanhos dos splits + hashes do bronze
```

| Camada | Escrita por | Contrato | Formato |
|---|---|---|---|
| bronze | download manual (Kaggle) | `validate_raw` (entrada de `load_bronze_domain`) | CSV |
| silver | `make silver` (`run_bronze_to_silver`) | `validate_processed` | Parquet |
| gold | `make gold` (`run_silver_to_gold`) | `validate_processed` | Parquet |

> `bronze/concatenated.csv` é o arquivo agregado do Kaggle e **não é usado** pelo
> pipeline; pode ser removido para economizar ~1,5 GB.

Fonte do bronze: [Kaggle — ptbr-sentiment-analysis-datasets](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets).
Detalhes de composição, licença e LGPD: [Datasheet](../reports/datasheets/ptbr_sentiment_datasets.md).
