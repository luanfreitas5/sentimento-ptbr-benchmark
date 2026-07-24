"""Carrega e valida a configuração do projeto com Pydantic.

Une ``configs/config.yaml`` e ``configs/model_params.yaml`` em um objeto
tipado. Uma configuração inválida falha no *startup* com um erro claro — nunca
no meio de um treino longo.

Examples
--------
>>> from src.config.settings import load_settings
>>> settings = load_settings()
>>> settings.project.random_seed
42
>>> settings.model_params.tfidf_logreg.logreg.C > 0
True
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import CONFIGS_DIR

CONFIG_FILE: Path = CONFIGS_DIR / "config.yaml"
MODEL_PARAMS_FILE: Path = CONFIGS_DIR / "model_params.yaml"


# --------------------------------------------------------------------------- #
# config.yaml
# --------------------------------------------------------------------------- #
class SamplingConfig(BaseModel):
    """Parâmetros de amostragem estratificada por domínio."""

    sample_size: int | None = Field(default=20_000, ge=1)
    stratify_by: str = "polarity"
    min_chars: int = Field(default=3, ge=0)


class SplitConfig(BaseModel):
    """Parâmetros da divisão treino/teste dentro de cada domínio."""

    test_size: float = Field(default=0.2, gt=0, lt=1)
    stratify: bool = True


class ProjectConfig(BaseModel):
    """Configurações gerais do projeto (config.yaml)."""

    project_name: str = "sentimento-ptbr-benchmark"
    random_seed: int = 42
    datasets: list[str] = Field(min_length=1)
    target: str = "polarity"
    text_column: str = "review_text"
    sampling: SamplingConfig = SamplingConfig()
    split: SplitConfig = SplitConfig()


# --------------------------------------------------------------------------- #
# model_params.yaml
# --------------------------------------------------------------------------- #
class TfidfParams(BaseModel):
    """Hiperparâmetros do vetorizador TF-IDF."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = Field(default=5, ge=1)
    max_df: float = Field(default=0.9, gt=0, le=1)
    max_features: int | None = Field(default=50_000, ge=1)
    sublinear_tf: bool = True
    strip_accents: str | None = None


class LogRegParams(BaseModel):
    """Hiperparâmetros da Regressão Logística."""

    C: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=1000, ge=1)
    class_weight: str | None = "balanced"
    solver: str = "liblinear"


class TfidfLogRegConfig(BaseModel):
    """Configuração do baseline TF-IDF + Regressão Logística."""

    tfidf: TfidfParams = TfidfParams()
    logreg: LogRegParams = LogRegParams()


class BertTrainingParams(BaseModel):
    """Hiperparâmetros do fine-tuning do BERTimbau."""

    epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    fp16: bool = True
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    logging_steps: int = Field(default=50, ge=1)
    early_stopping_patience: int = Field(default=2, ge=0)


class BertimbauConfig(BaseModel):
    """Configuração do transformer BERTimbau."""

    pretrained_model: str = "neuralmind/bert-base-portuguese-cased"
    revision: str | None = Field(
        default=None,
        description=(
            "Commit SHA imutável do modelo/tokenizer no Hugging Face Hub. "
            "Recomendado em produção para evitar supply-chain attacks "
            "(CWE-494); None usa a revisão padrão ('main') do repositório."
        ),
    )
    max_length: int = Field(default=128, ge=8, le=512)
    num_labels: int = Field(default=2, ge=2)
    training: BertTrainingParams = BertTrainingParams()


class EvaluationConfig(BaseModel):
    """Parâmetros de avaliação (métrica principal, incerteza)."""

    primary_metric: str = "f1_macro"
    bootstrap_samples: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class ModelParamsConfig(BaseModel):
    """Raiz do model_params.yaml."""

    tfidf_logreg: TfidfLogRegConfig = TfidfLogRegConfig()
    bertimbau: BertimbauConfig = BertimbauConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# --------------------------------------------------------------------------- #
# Settings raiz (com segredos vindos do .env)
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Configuração global validada do projeto.

    Combina os YAMLs (``config.yaml`` + ``model_params.yaml``) com variáveis de
    ambiente/segredos carregados de ``.env`` (nunca commitados).

    Attributes
    ----------
    project : ProjectConfig
        Configurações gerais e de dados.
    model_params : ModelParamsConfig
        Hiperparâmetros dos modelos e da avaliação.
    mlflow_tracking_uri : str
        URI do backend do MLflow (default: store local ``./mlruns``).
    hf_token : str | None
        Token do Hugging Face (opcional; só para modelos privados).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTIBENCH_",
        extra="ignore",
        protected_namespaces=(),
    )

    project: ProjectConfig
    model_params: ModelParamsConfig
    mlflow_tracking_uri: str = "./mlruns"
    hf_token: str | None = None


def _read_yaml(path: Path) -> dict:
    """Lê um arquivo YAML e retorna um dicionário; falha se ausente."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_settings(
    config_file: Path = CONFIG_FILE,
    model_params_file: Path = MODEL_PARAMS_FILE,
) -> Settings:
    """Carrega, mescla e valida toda a configuração do projeto.

    Parameters
    ----------
    config_file : Path, optional
        Caminho do ``config.yaml``.
    model_params_file : Path, optional
        Caminho do ``model_params.yaml``.

    Returns
    -------
    Settings
        Configuração validada e tipada.

    Raises
    ------
    FileNotFoundError
        Se algum YAML de configuração estiver ausente.
    pydantic.ValidationError
        Se algum valor violar as restrições declaradas.

    Examples
    --------
    >>> settings = load_settings()
    >>> "b2w" in settings.project.datasets
    True
    """
    project = ProjectConfig(**_read_yaml(config_file))
    model_params = ModelParamsConfig(**_read_yaml(model_params_file))
    return Settings(project=project, model_params=model_params)
