# Datasheet — Brazilian Portuguese Sentiment Analysis Datasets

Estrutura: *Datasheets for Datasets* (Gebru et al., 2021). Versão dos dados descrita: CSVs bronze identificados pelos hashes da §4.

## 1. Motivação
- Por que foi criado? Para qual tarefa? O conjunto reúne cinco corpora públicos de avaliações em português brasileiro, rotulados em polaridade, para classificação de sentimento. Neste projeto serve para comparar TF-IDF + Regressão Logística e BERTimbau e medir a generalização entre domínios (treina em um corpus, testa nos outros).
- Quem criou e quem financiou? Os corpora originais são de terceiros (B2W/Americanas, Buscapé, Olist e UTLC — apps e filmes); a agregação é de [fredericods no Kaggle](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets). TODO(owner): registrar autores e financiamento de cada corpus original.

## 2. Composição
- O que cada instância representa? Uma avaliação textual escrita por um usuário sobre um produto, app ou filme.
- Número de instâncias e período coberto: os cinco domínios somam milhões de linhas no bronze (`concatenated.csv` ~1,5 GB); o projeto usa uma amostra estratificada de 20.000 por domínio (~16.000 treino / 4.000 teste após deduplicação). Período coberto: 2026.
- Campos, tipos e contrato: `review_text` (texto), `polarity` (0/1), `rating` (1–5); contratos `validate_raw` (bronze) e `validate_processed` (silver/gold) em `src/schemas/dataset.py`. As colunas originais adicionais (texto pré-processado/tokenizado, índices de k-fold) não são usadas.
- Rótulo: `polarity` binária (0 = negativo, 1 = positivo), derivada da nota na origem; reviews neutras foram descartadas na construção original. TODO(owner): confirmar o corte de nota por corpus.
- Dados ausentes (taxa por coluna): linhas com nulo em texto, polaridade ou nota são removidas no silver; TODO(owner): registrar a taxa por coluna e por domínio.
- Campos sensíveis / PII e tratamento: não há identificadores diretos por design; o texto livre pode conter, ocasionalmente, nomes, telefones ou e-mails digitados pelo autor — TODO(owner): medir a incidência. Nada de texto de usuário é registrado em logs.
- É amostra de uma população maior? Como amostrado? Sim: estratificada por polaridade (seed 42), `sample_size: 20000` em `configs/config.yaml`; mantém a proporção de classes. Prevalência de positivos no teste: b2w 69,7% · buscape 90,1% · olist 66,6% · utlc_apps 71,7% · utlc_movies 88,3%.

## 3. Processo de coleta
- Fonte(s) e sistema(s) de origem: avaliações públicas coletadas das respectivas plataformas pelos autores originais; este projeto não coleta dados novos.
- Período e frequência de coleta: TODO(owner): não informado pela fonte.
- Consentimento / aviso aos titulares: TODO(owner): não documentado pela fonte; tratar como opiniões públicas de terceiros.

## 4. Pré-processamento
- Limpeza, filtragem e linhas removidas por regra: remoção de nulos; remoção de textos com menos de `min_chars` (3) caracteres após a limpeza; limpeza leve para o TF-IDF (minúsculas, URLs/menções, repetições; acentos preservados; o BERTimbau usa o texto original); deduplicação por texto limpo; amostragem e split estratificados. TODO(owner): contagem de linhas removidas por regra (não registrada).
- Anonimização / pseudonimização: não aplicada (sem identificadores diretos); não há segredo/sal.
- Hash SHA-256 dos dados brutos (`data/bronze/manifest.json`):
  - `b2w.csv` `fc51e0ff18eed320fd4788c81a726402d20cb356cb66cd92dbd60c28a62a9d7c`
  - `buscape.csv` `4b38ebd29a3a07f8970f26502aadd6e782139a6cd00a8a739da0551dd64c032b`
  - `olist.csv` `879a5a7ef489b68a89ceda5985ef0256646edd0de9537208f202d13d98842dd6`
  - `utlc_apps.csv` `162b1a83effe5ef0897fad521b73c0a9fa58231371f8ba90d29d71d329f24218`
  - `utlc_movies.csv` `b64a1ccc8eaad5dc01feb1939927f853d9d94e32453b9a6052514fc7db8adb1b`
- Camadas: `data/bronze` (imutável) → `data/silver` (parquet validado; `manifest.json` com linhas e hash) → `data/gold` (splits `train`/`test` por domínio). Ver [`data/README.md`](../../data/README.md).

## 5. Usos
- Usos pretendidos: pesquisa e portfólio em classificação de sentimento PT-BR e em generalização entre domínios.
- Usos desaconselhados: inferência sobre pessoas identificáveis; decisões automatizadas de alto impacto; extrapolar a outros gêneros textuais (notícias, redes sociais).
- Riscos de viés conhecidos: forte desbalanceamento para a classe positiva (até 90%); viés das plataformas e de época; domínios com estilos muito distintos (filmes vs. varejo) derrubam a transferência.

## 6. Distribuição
- Licença e restrições: cada corpus tem sua própria licença — consulte a página do Kaggle antes de redistribuir. Este repositório **não** redistribui os dados (`data/` fora do Git). TODO(owner): transcrever a licença de cada corpus.
- Onde está armazenado (DVC remote, bucket): somente local (`data/`); sem remote DVC configurado.

## 7. Manutenção
- Responsável: Luan Freitas — luan.mgf@gmail.com
- Frequência de atualização: sob demanda; qualquer mudança nos CSVs altera os hashes do `manifest.json`.
- Como atender pedidos de exclusão de titulares: remover as linhas correspondentes do bronze e regerar silver/gold com `make clean-processed data`; o texto pode ser localizado por busca textual no bronze.

## 8. LGPD
- Base legal (art. 7 / art. 11): TODO(owner): definir com o encarregado; para pesquisa/portfólio com dados públicos, avaliar o art. 7º, IV (estudos por órgão de pesquisa) ou o legítimo interesse (art. 7º, IX).
- Prazo de retenção: enquanto durar o projeto; TODO(owner): definir.
- Relatório de impacto (RIPD) necessário? Não, enquanto não houver dados sensíveis nem implantação; reavaliar se PII for detectada nos textos.
