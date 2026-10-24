# =============================================================================
# Makefile — atalhos reprodutíveis. `make check` replica o CI localmente.
# =============================================================================
UV := uv
RUN := $(UV) run python src/main.py    # 'src' vira raiz do path ao rodar o script
DOCKER_RUN := docker compose run --rm sentibench

.DEFAULT_GOAL := help
.PHONY: help install install-bert install-all hooks update lock \
	lint format typecheck security quality test smoke check pre-commit update-hooks release \
	data prepare silver gold baseline bert benchmark evaluate train reproduce mlflow-ui \
	clean-processed clean-reports clean-outputs clean-notebooks \
	docker-build docker-build-bert docker-prepare docker-benchmark docker-evaluate docker-mlflow-ui docker-lint \n	docs docs-serve docs-deploy profile clean

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------
install:  ## Instala dependências (runtime + dev) e os hooks do pre-commit
	$(UV) sync --locked
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

install-bert:  ## Instala o extra do BERTimbau (torch + transformers)
	$(UV) sync --locked --extra bert

install-all:  ## Instala todos os extras + grupos
	$(UV) sync --locked --all-extras --all-groups

hooks:  ## Instala os hooks do pre-commit
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

update:  ## Atualiza as dependências e regenera o uv.lock
	$(UV) lock --upgrade
	$(UV) sync --all-groups

lock:  ## Regenera o uv.lock
	$(UV) lock

# --- Qualidade --------------------------------------------------------------
lint:  ## Ruff: lint + verificação de formatação
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:  ## Corrige lint e formata o código com ruff
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck:  ## Type checking estático (basedpyright)
	$(UV) run basedpyright

security:  ## Segurança: bandit + pip-audit
	$(UV) run bandit -c pyproject.toml -r src
	$(UV) run pip-audit --ignore-vuln PYSEC-2026-2447  # diskcache (via dvc): sem versão corrigida

quality:  ## Dead code, modernização, complexidade e docstrings
	$(UV) run vulture src
	$(UV) run refurb src
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src
	$(UV) run interrogate -v src

test:  ## Testes com cobertura (>= 80%)
	$(UV) run pytest -m "not slow" --cov-report=term-missing

smoke:  ## Apenas os smoke tests (os mesmos do pre-commit)
	$(UV) run pytest -m smoke -q --no-cov

check: lint typecheck security quality test  ## Suíte completa (espelha o CI)

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza as revisões dos hooks do pre-commit
	$(UV) run pre-commit autoupdate

release:  ## Cria uma nova release (versão + changelog + tag)
	$(UV) run cz bump --changelog --yes

# --- Pipeline (bronze -> silver -> gold -> modelos -> relatórios) -----------
# Pré-requisito: CSVs do Kaggle em data/bronze/ (ver data/README.md).
silver:  ## bronze -> silver: limpa, valida (pandera) e grava parquet + manifest
	$(RUN) bronze-to-silver

gold:  ## silver -> gold: amostra estratificada e split treino/teste
	$(RUN) silver-to-gold

prepare data:  ## bronze -> silver -> gold
	$(RUN) prepare

baseline:  ## Treina o baseline TF-IDF + Regressão Logística em todos os domínios
	$(RUN) train --model tfidf_logreg

bert:  ## Fine-tuning do BERTimbau em todos os domínios (requer install-bert)
	$(RUN) train --model bertimbau

benchmark:  ## Benchmark cross-dataset completo (treina N x testa N)
	$(RUN) benchmark

evaluate:  ## Consolida métricas, significância e figuras
	$(RUN) evaluate

train: baseline  ## Alias do modelo base (TF-IDF + LogReg); use `make bert` para o BERTimbau

reproduce: data benchmark evaluate  ## Reproduz todos os resultados de reports/

mlflow-ui:  ## UI local do MLflow (backend SQLite)
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# --- Docker (imagem = código + uv.lock; dados/modelos/mlruns via volumes) ---
docker-build:  ## Constrói a imagem (baseline TF-IDF + LogReg, CPU)
	docker compose build sentibench

docker-build-bert:  ## Constrói a imagem com o extra do BERTimbau (torch + transformers)
	UV_EXTRAS="--extra bert" docker compose build sentibench

docker-prepare:  ## Container: bronze -> silver -> gold
	$(DOCKER_RUN) prepare

docker-benchmark:  ## Container: benchmark cross-dataset completo
	$(DOCKER_RUN) benchmark

docker-evaluate:  ## Container: consolida métricas, significância e figuras
	$(DOCKER_RUN) evaluate

docker-mlflow-ui:  ## UI do MLflow em http://localhost:5000 (container)
	docker compose up mlflow-ui

docker-lint:  ## Valida o compose e o Dockerfile (hadolint)
	docker compose config -q
	hadolint Dockerfile

# --- Limpeza de saídas do pipeline ------------------------------------------
clean-processed:  ## Remove os artefatos das camadas silver e gold
	rm -rf data/silver/*.parquet data/silver/manifest.json data/gold/*/*.parquet data/gold/manifest.json

clean-reports:  ## Remove os relatórios gerados (pastas por modelo + comparação)
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

clean-outputs: clean-processed clean-reports  ## Remove todas as saídas do pipeline

clean-notebooks:  ## Remove as saídas dos notebooks (nbstripout)
	$(UV) run nbstripout notebooks/*.ipynb

# --- Documentação -----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

# --- Utilitários ------------------------------------------------------------
profile:  ## Profiling com scalene (ajuste o alvo)
	$(UV) run scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml site
	find . -type d -name __pycache__ -exec rm -rf {} +
