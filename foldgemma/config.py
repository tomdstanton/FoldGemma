from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    """Model architecture variants for FoldGemma."""

    FOLDGEMMA = "foldgemma"
    FOLDGEMMA_T5 = "foldgemma_t5"


@dataclass(frozen=True, slots=True)
class FoldGemmaConfig:
    """Unified configuration for FoldGemma model."""

    vocab_size: int = 64
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    model_type: ModelType = ModelType.FOLDGEMMA

    def __post_init__(self) -> None:
        if isinstance(self.model_type, str) and not isinstance(self.model_type, ModelType):
            object.__setattr__(self, "model_type", ModelType(self.model_type))

