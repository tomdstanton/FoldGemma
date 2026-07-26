"""PyTorch implementation of Gemma Bidirectional Encoder for FoldGemma."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldgemma.config import FoldGemmaConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embeddings to input query or key tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos: Cosine frequencies of shape (seq_len, head_dim).
        sin: Sine frequencies of shape (seq_len, head_dim).

    Returns:
        RoPE-embedded tensor of shape (batch, seq_len, num_heads, head_dim).
    """
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed * self.scale


class GemmaMLP(nn.Module):
    """GeGLU MLP block for Gemma."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = F.gelu(gate, approximate="tanh") * up
        return self.down_proj(activated)


class GemmaAttention(nn.Module):
    """Grouped Query Attention (GQA) module with RoPE and bidirectional attention."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads_per_group = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def _compute_rope(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(0, self.head_dim, device=device, dtype=torch.float32)[::2]
                / self.head_dim
            )
        )
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = self._compute_rope(seq_len, x.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_heads_per_group, dim=2)
        v = v.repeat_interleave(self.num_heads_per_group, dim=2)

        # Transpose to (batch, num_heads, seq_len, head_dim) for F.scaled_dot_product_attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Attention calculation (causal or bidirectional)
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out)


class GemmaDecoderLayer(nn.Module):
    """Transformer decoder block for Gemma."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x), is_causal=is_causal)
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class GemmaModel(nn.Module):
    """Gemma Bidirectional Encoder with sequence classification head in PyTorch."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def __getattr__(name: str):
    """Dynamic module attribute lookup for backward-compatible re-exports."""
    if name == "BaseFoldModel":
        from foldgemma.models.base import BaseFoldModel

        return BaseFoldModel
    if name == "FoldGemma":
        from foldgemma.models.foldgemma import FoldGemma

        return FoldGemma
    if name == "FoldGemmaT5":
        from foldgemma.models.foldgemma_t5 import FoldGemmaT5

        return FoldGemmaT5
    if name == "GemmaCrossAttention":
        from foldgemma.models.foldgemma_t5 import GemmaCrossAttention

        return GemmaCrossAttention
    if name == "GemmaT5DecoderLayer":
        from foldgemma.models.foldgemma_t5 import GemmaT5DecoderLayer

        return GemmaT5DecoderLayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
