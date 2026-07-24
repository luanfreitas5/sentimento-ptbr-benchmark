"""Fine-tuning do BERTimbau para classificação de sentimento.

Envolve o ``neuralmind/bert-base-portuguese-cased`` (BERTimbau) atrás da
interface ``SentimentClassifier``. As dependências pesadas (torch, transformers)
são importadas de forma preguiçosa: o módulo pode ser importado sem o extra
``bert`` instalado — só a instanciação/uso é que as exige.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.config.settings import BertimbauConfig
from src.exceptions.model import MissingDependencyError, ModelNotFittedError
from src.models.base import SentimentClassifier

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # apenas para type checking; não executa em runtime.
    import torch  # pyright: ignore[reportMissingImports]


def _require_transformers() -> tuple[Any, Any]:
    """Importa torch e transformers ou levanta erro claro se ausentes.

    Returns
    -------
    tuple[Any, Any]
        Módulos ``torch`` e ``transformers``.

    Raises
    ------
    MissingDependencyError
        Se o extra ``bert`` (torch/transformers) não estiver instalado.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
        import transformers  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # dependência opcional ausente
        raise MissingDependencyError(
            "O BERTimbau requer o extra 'bert'. Instale com: uv sync --extra bert"
        ) from exc
    return torch, transformers


class _TextDataset:
    """Dataset PyTorch mínimo com textos já tokenizados e (opcionalmente) rótulos."""

    def __init__(self, encodings: dict[str, Any], labels: Sequence[int] | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import torch  # pyright: ignore[reportMissingImports]

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class BertimbauClassifier(SentimentClassifier):
    """Classificador de sentimento por fine-tuning do BERTimbau.

    Parameters
    ----------
    config : BertimbauConfig
        Configuração validada (modelo pré-treinado, ``max_length``, treino).
    seed : int, optional
        Semente de reprodutibilidade, by default 42.
    output_dir : Path | None, optional
        Diretório de trabalho para checkpoints do ``Trainer``; se ``None``, usa
        um subdiretório temporário ao treinar.

    Notes
    -----
    Requer o extra ``bert`` (``uv sync --extra bert``).
    """

    name: ClassVar[str] = "bertimbau"

    def __init__(
        self,
        config: BertimbauConfig,
        *,
        seed: int = 42,
        output_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.output_dir = output_dir
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------ #
    # Treino
    # ------------------------------------------------------------------ #
    def _load_tokenizer(self) -> Any:
        """Carrega o tokenizer do BERTimbau (cacheado na instância)."""
        _, transformers = _require_transformers()
        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.pretrained_model, revision=self.config.revision
            )
        return self._tokenizer

    def _encode(self, texts: Sequence[str]) -> dict[str, Any]:
        """Tokeniza uma sequência de textos com truncamento/padding."""
        tokenizer = self._load_tokenizer()
        return dict(
            tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
            )
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> BertimbauClassifier:
        """Faz o fine-tuning do BERTimbau nos textos e rótulos fornecidos.

        Parameters
        ----------
        texts : Sequence[str]
            Textos de treino (recomenda-se o texto original, não o limpo).
        labels : Sequence[int]
            Rótulos de polaridade (0/1).

        Returns
        -------
        BertimbauClassifier
            O próprio modelo, com pesos ajustados.
        """

        torch, transformers = _require_transformers()
        transformers.set_seed(self.seed)

        logger.info(
            "Fine-tuning do BERTimbau (%s) em %d exemplos",
            self.config.pretrained_model,
            len(texts),
        )

        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.config.pretrained_model,
            num_labels=self.config.num_labels,
            revision=self.config.revision,
        )

        train_dataset = _TextDataset(self._encode(texts), labels)
        train_cfg = self.config.training
        work_dir = self.output_dir or Path(tempfile.mkdtemp(prefix="bertimbau_"))

        args = transformers.TrainingArguments(
            output_dir=str(work_dir),
            num_train_epochs=train_cfg.epochs,
            per_device_train_batch_size=train_cfg.batch_size,
            per_device_eval_batch_size=train_cfg.eval_batch_size,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            warmup_ratio=train_cfg.warmup_ratio,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            fp16=train_cfg.fp16 and torch.cuda.is_available(),
            logging_steps=train_cfg.logging_steps,
            report_to=[],  # o rastreamento com MLflow é feito no pipeline
            seed=self.seed,
            save_strategy="no",
        )

        trainer = transformers.Trainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            data_collator=transformers.DataCollatorWithPadding(self._load_tokenizer()),
        )
        trainer.train()
        self._model.eval()
        return self

    # ------------------------------------------------------------------ #
    # Inferência
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> Any:
        """Garante que o modelo foi treinado/carregado antes de inferir."""
        if self._model is None:
            raise ModelNotFittedError("O BERTimbau ainda não foi treinado nem carregado.")
        return self._model

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz probabilidades por classe (softmax do logit).

        Returns
        -------
        np.ndarray
            Matriz ``(n, 2)`` na ordem ``[negativo, positivo]``.
        """
        torch, _ = _require_transformers()
        model = self._check_fitted()
        device = next(model.parameters()).device

        batch_size = self.config.training.eval_batch_size
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = list(texts)[start : start + batch_size]
                encodings = self._encode(chunk)
                inputs = {key: torch.tensor(val).to(device) for key, val in encodings.items()}
                logits = model(**inputs).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz os rótulos de polaridade (argmax das probabilidades)."""
        return self.predict_proba(texts).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """Salva pesos e tokenizer no formato nativo do transformers."""
        model = self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        self._load_tokenizer().save_pretrained(path)
        logger.info("BERTimbau salvo em %s", path)

    @classmethod
    def load(cls, path: Path, *, config: BertimbauConfig | None = None) -> BertimbauClassifier:
        """Carrega um BERTimbau ajustado salvo com ``save``.

        Parameters
        ----------
        path : Path
            Diretório com os pesos e o tokenizer.
        config : BertimbauConfig | None, optional
            Configuração; se ``None``, usa os defaults apontando para ``path``.

        Returns
        -------
        BertimbauClassifier
            Instância pronta para inferência.
        """
        _, transformers = _require_transformers()
        path = Path(path)
        cfg = config or BertimbauConfig(pretrained_model=str(path))
        model = cls(cfg)
        # path é um diretório local (checkpoint salvo por `save`), não um
        # repo_id do Hub — não há revisão remota para pinar aqui.
        model._model = transformers.AutoModelForSequenceClassification.from_pretrained(path)  # nosec B615
        model._model.eval()
        model._tokenizer = transformers.AutoTokenizer.from_pretrained(path)  # nosec B615
        return model
