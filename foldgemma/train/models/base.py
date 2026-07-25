"""JAX/Flax abstract base class model for FoldGemma."""

import math

from flax import nnx
import jax
import jax.numpy as jnp

from foldgemma.config import FoldGemmaConfig
from foldgemma.train.models.gemma import GemmaDecoderLayer, RMSNorm


class BaseFoldModel(nnx.Module):
    """Abstract base class in Flax handling token embeddings, RoPE, bidirectional Gemma blocks,
    RMSNorm, and pLDDT quality score mask ingestion.
    """

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs
        )
        self.layers = [
            GemmaDecoderLayer(config, rngs=rngs)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def encode(
        self, input_ids: jax.Array, plddt: jax.Array | None = None, plddt_threshold: float = 70.0
    ) -> jax.Array:
        """Encodes input token IDs through embedding, Gemma decoder layers, norm, and pLDDT masking."""
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if plddt is not None:
            mask = (plddt >= plddt_threshold).astype(x.dtype)
            x = x * jnp.expand_dims(mask, axis=-1)
        return x

    def __call__(
        self,
        input_ids: jax.Array,
        decoder_input_ids: jax.Array | None = None,
        plddt: jax.Array | None = None,
        plddt_threshold: float = 70.0,
    ) -> jax.Array:
        """Default forward pass calls encode."""
        return self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
