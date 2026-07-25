"""Flax FoldGemma sequence classification model."""

from flax import nnx
import jax

from foldgemma.train.models.base import BaseFoldModel
from foldgemma.config import FoldGemmaConfig

class FoldGemma(BaseFoldModel):
    """Flax FoldGemma model with dense sequence classification head (lm_head)."""

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        decoder_input_ids: jax.Array | None = None,
        plddt: jax.Array | None = None,
        plddt_threshold: float = 70.0,
    ) -> jax.Array:
        """Forward pass running BaseFoldModel encoder followed by classification lm_head.

        Returns logits of shape (batch, seq_len, vocab_size).
        """
        hidden_states = self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
        return self.lm_head(hidden_states)
