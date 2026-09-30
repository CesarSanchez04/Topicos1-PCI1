"""
Esquema de salida estructurada para el Agente Auditor de Citas Cientificas.
Semanas 3 CC-0F4: JSON Schema & Pydantic Validation.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RelationEnum(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    NEUTRAL = "neutral"


class CitationRelationOutput(BaseModel):
    relation: RelationEnum = Field(
        ...,
        description="Relacion clasificada entre las dos afirmaciones cientificas: agree, disagree o neutral."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza asignado por el modelo entre 0.0 y 1.0."
    )
    reason: str = Field(
        ...,
        min_length=5,
        description="Justificacion explicativa breve del desacuerdo o acuerdo."
    )
    citation_valid: Optional[bool] = Field(
        default=True,
        description="Flag booleano indicando si el contexto de cita provee evidencia directa."
    )

    @field_validator("confidence", mode="before")
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v


def get_json_schema() -> dict:
    return CitationRelationOutput.model_json_schema()


if __name__ == "__main__":
    import json
    print("JSON Schema exportado:")
    print(json.dumps(get_json_schema(), indent=2))
