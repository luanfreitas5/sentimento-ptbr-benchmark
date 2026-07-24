"""Contratos de dados (pandera) por estágio do pipeline.

Validam os DataFrames na entrada e na saída de cada etapa, falhando cedo com
um erro claro em vez de propagar corrupção silenciosa.

Módulos
-------
dataset
    Esquemas do corpus bruto e do dataset processado pronto para modelagem.
"""

from src.schemas.dataset import ProcessedSchema, RawSchema, validate_processed, validate_raw

__all__ = ["ProcessedSchema", "RawSchema", "validate_processed", "validate_raw"]
