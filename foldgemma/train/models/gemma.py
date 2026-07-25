"""JAX/Flax NNX implementation of Gemma Bidirectional Encoder for FoldGemma."""

import math

from flax import nnx
import jax
import jax.numpy as jnp

from foldgemma.config import FoldGemmaConfig


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    cos = jnp.expand_dims(jnp.expand_dims(cos, axis=0), axis=2)
    sin = jnp.expand_dims(jnp.expand_dims(sin, axis=0), axis=2)
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.eps)
        return normed * self.scale.value


class GemmaMLP(nnx.Module):
    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = jax.nn.gelu(gate) * up
        return self.down_proj(activated)


class GemmaAttention(nnx.Module):
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

        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.reshape((batch, seq_len, self.num_heads * self.head_dim))
        return self.o_proj(attn_out)


class GemmaDecoderLayer(nnx.Module):
    def __init__(self, config: FoldGemmaConfig, rngs: nnx.Rngs):
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x))
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class GemmaModel(nnx.Module):
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
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def __getattr__(name: str):
    if name == "BaseFoldModel":
        from foldgemma.train.models.base import BaseFoldModel
        return BaseFoldModel
    if name == "FoldGemma":
        from foldgemma.train.models.foldgemma import FoldGemma
        return FoldGemma
    if name == "FoldGemmaT5":
        from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5
        return FoldGemmaT5
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
