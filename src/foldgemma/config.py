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

    @classmethod
    def small(cls, model_type: ModelType = ModelType.FOLDGEMMA) -> "FoldGemmaConfig":
        """Small variant (Default): ~10M parameters. Perfect for rapid prototyping."""
        return cls(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            model_type=model_type,
        )

    @classmethod
    def base(cls, model_type: ModelType = ModelType.FOLDGEMMA) -> "FoldGemmaConfig":
        """Base variant: ~100M parameters. Standard for high-quality mappings."""
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            model_type=model_type,
        )

    @classmethod
    def large(cls, model_type: ModelType = ModelType.FOLDGEMMA) -> "FoldGemmaConfig":
        """Large variant: ~350M parameters. For state-of-the-art accuracy."""
        return cls(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            model_type=model_type,
        )

