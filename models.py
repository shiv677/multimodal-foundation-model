from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=256, ge=16)
    image_size: int = Field(default=32, ge=8)
    patch_size: int = Field(default=8, ge=2)
    vision_dim: int = Field(default=64, ge=16)
    vision_layers: int = Field(default=2, ge=1)
    vision_heads: int = Field(default=4, ge=1)
    text_dim: int = Field(default=64, ge=16)
    text_layers: int = Field(default=2, ge=1)
    text_heads: int = Field(default=4, ge=1)
    mlp_dim: int = Field(default=128, ge=16)
    max_seq_len: int = Field(default=256, ge=16)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "ModelConfig":
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.vision_dim % self.vision_heads != 0:
            raise ValueError("vision_dim must be divisible by vision_heads")
        if self.text_dim % self.text_heads != 0:
            raise ValueError("text_dim must be divisible by text_heads")
        return self


class VisualReasoningSample(BaseModel):
    id: str
    question: str
    options: List[str] = Field(default_factory=list)
    ground_truth: str
    image_meta: Dict[str, Any] = Field(default_factory=dict)


class InferenceOutput(BaseModel):
    generated_text: str
    generated_token_ids: List[int]
    latency_ms: float


class EvaluationResult(BaseModel):
    total_samples: int
    correct_count: int
    accuracy: float
    avg_latency_ms: float
    predictions: List[Dict[str, Any]] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    device: str
    param_count: int
    eval_metrics: EvaluationResult
    throughput_fps: float
