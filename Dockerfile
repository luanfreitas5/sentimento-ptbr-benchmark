# syntax=docker/dockerfile:1.7
# Imagem do benchmark (job em lote via CLI): código + dependências travadas.
# Dados, modelos, relatórios, logs e o banco do MLflow NÃO entram aqui: são
# montados como volumes em tempo de execução (ver exemplo de uso abaixo).
# Fixe as tags por patch; em produção acrescente @sha256:<digest>
# (obtido com `docker buildx imagetools inspect <imagem>:<tag>`). Mesma tag nos dois estágios.
#
# Build (baseline TF-IDF + LogReg, CPU, imagem enxuta):
#   docker build -t sentimento-ptbr-benchmark .
# Build com o extra do BERTimbau (torch + transformers; imagem de vários GB):
#   docker build --build-arg UV_EXTRAS="--extra bert" -t sentimento-ptbr-benchmark:bert .
# Execução:
#   docker run --rm -v ./data:/app/data -v ./models:/app/models -v ./reports:/app/reports \
#       -v ./mlruns:/app/mlruns -v ./logs:/app/logs sentimento-ptbr-benchmark benchmark

# ---------- Estágio 1: binário do uv, versão fixa ----------
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

# ---------- Estágio 2: resolve o ambiente a partir do uv.lock ----------
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
ARG UV_EXTRAS=""
WORKDIR /app
# Só os manifestos: esta camada fica em cache até o lock mudar.
# --locked falha se o uv.lock estiver desatualizado em relação ao pyproject.toml.
# --no-dev mantém fora as ferramentas de desenvolvimento.
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project ${UV_EXTRAS}

# ---------- Estágio 3: runtime enxuto, sem uv, sem compiladores ----------
FROM python:3.12.11-slim-bookworm AS runtime
# libgomp1: runtime OpenMP exigido por scikit-learn/torch; ausente na imagem slim.
# A versão do pacote segue a da imagem base fixada acima.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
# Código e venv pertencem ao root: o usuário da aplicação só lê
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY configs/ configs/
# Diretórios de saída (montados como volumes) e caches: únicos pontos graváveis pelo usuário app
RUN mkdir -p data models reports logs mlruns /cache/hf /cache/mpl \
    && chown -R app:app data models reports logs mlruns /cache
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME="/cache/hf" \
    MPLCONFIGDIR="/cache/mpl" \
    SENTIBENCH_MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db"
USER app:app
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
