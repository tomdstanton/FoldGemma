"""Flax FoldGemmaT5 encoder-decoder architecture with cross-attention and generation."""

import math
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp

from foldgemma.config import FoldGemmaConfig
from foldgemma.train.models.base import BaseFoldModel
from foldgemma.train.models.gemma import GemmaMLP, RMSNorm, apply_rope


class GemmaCausalSelfAttention(nnx.Module):
    """Grouped Query Causal Self-Attention module with RoPE."""

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads_per_group = self.num_heads // self.num_kv_heads

        self.q_proj = nnx.Linear(config.hidden_size, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(self.num_heads * self.head_dim, config.hidden_size, use_bias=False, rngs=rngs)

    def _compute_rope(self, seq_len: int) -> tuple[jax.Array, jax.Array]:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)
        )
        pos = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(pos, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape((batch, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch, seq_len, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape((batch, seq_len, self.num_kv_heads, self.head_dim))

        cos, sin = self._compute_rope(seq_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = jnp.repeat(k, self.num_heads_per_group, axis=2)
        v = jnp.repeat(v, self.num_heads_per_group, axis=2)

        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.reshape((batch, seq_len, self.num_heads * self.head_dim))
        return self.o_proj(attn_out)


class GemmaCrossAttention(nnx.Module):
    """Grouped Query Cross-Attention module for decoder queries over encoder keys/values."""

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads_per_group = self.num_heads // self.num_kv_heads

        self.q_proj = nnx.Linear(config.hidden_size, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(self.num_heads * self.head_dim, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, encoder_hidden_states: jax.Array) -> jax.Array:
        batch, tgt_len, _ = x.shape
        src_len = encoder_hidden_states.shape[1]

        q = self.q_proj(x).reshape((batch, tgt_len, self.num_heads, self.head_dim))
        k = self.k_proj(encoder_hidden_states).reshape(
            (batch, src_len, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(encoder_hidden_states).reshape(
            (batch, src_len, self.num_kv_heads, self.head_dim)
        )

        k = jnp.repeat(k, self.num_heads_per_group, axis=2)
        v = jnp.repeat(v, self.num_heads_per_group, axis=2)

        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.reshape((batch, tgt_len, self.num_heads * self.head_dim))
        return self.o_proj(attn_out)


class GemmaT5DecoderLayer(nnx.Module):
    """T5 Decoder Layer combining Causal Self Attention, Cross Attention, and MLP."""

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.self_attn = GemmaCausalSelfAttention(config, rngs=rngs)
        self.cross_attn = GemmaCrossAttention(config, rngs=rngs)
        self.mlp = GemmaMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: jax.Array, encoder_hidden_states: jax.Array) -> jax.Array:
        residual = x
        x = self.self_attn(self.input_layernorm(x))
        x = residual + x

        residual = x
        x = self.cross_attn(self.post_attention_layernorm(x), encoder_hidden_states)
        x = residual + x

        residual = x
        x = self.mlp(self.pre_feedforward_layernorm(x))
        return residual + x


class FoldGemmaT5(BaseFoldModel):
    """Flax FoldGemmaT5 encoder-decoder architecture with cross-attention and generation."""

    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.decoder_embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs
        )
        self.decoder_layers = [
            GemmaT5DecoderLayer(config, rngs=rngs)
            for _ in range(config.num_hidden_layers)
        ]
        self.decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        decoder_input_ids: jax.Array | None = None,
        plddt: jax.Array | None = None,
        plddt_threshold: float = 70.0,
    ) -> jax.Array:
        """Forward pass for FoldGemmaT5 sequence-to-sequence model.

        Returns decoder logits of shape (batch, decoder_seq_len, vocab_size).
        """
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        encoder_hidden_states = self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
        x = self.decoder_embed_tokens(decoder_input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.decoder_layers:
            x = layer(x, encoder_hidden_states)
        x = self.decoder_norm(x)
        return self.lm_head(x)

    def generate(
        self,
        input_ids: jax.Array,
        plddt: jax.Array | None = None,
        plddt_threshold: float = 70.0,
        max_new_tokens: int = 32,
        bos_token_id: int = 2,
        eos_token_id: int | None = 1,
    ) -> jax.Array:
        """Autoregressively generates target tokens from input sequence and optional pLDDT."""
        
        # Note: In NNX, the model object is stateful, so we do not need to pass params dict 
        # around via `self.apply` like in flax.linen. We can directly call the methods.
        encoder_hidden_states = self.encode(
            input_ids, plddt=plddt, plddt_threshold=plddt_threshold
        )

        batch_size = input_ids.shape[0]
        ys = jnp.full((batch_size, 1), bos_token_id, dtype=jnp.int32)

        for _ in range(max_new_tokens):
            dec_x = self.decoder_embed_tokens(ys) * math.sqrt(self.config.hidden_size)
            for layer in self.decoder_layers:
                dec_x = layer(dec_x, encoder_hidden_states)
            dec_x = self.decoder_norm(dec_x)
            logits = self.lm_head(dec_x)
            next_tokens = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            ys = jnp.concatenate([ys, next_tokens], axis=1)
            if eos_token_id is not None and jnp.all(jnp.any(ys == eos_token_id, axis=-1)):
                break

        return ys
