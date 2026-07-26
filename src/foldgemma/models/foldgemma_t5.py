"""PyTorch implementation of FoldGemmaT5 sequence-to-sequence model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldgemma.config import FoldGemmaConfig
from foldgemma.models.base import BaseFoldModel
from foldgemma.models.gemma import GemmaAttention, GemmaMLP, RMSNorm


class GemmaCrossAttention(nn.Module):
    """Grouped Query Cross Attention with queries from decoder and keys/values from encoder."""

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

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for cross-attention.

        Args:
            hidden_states: Decoder hidden states of shape (batch, tgt_len, hidden_size).
            encoder_hidden_states: Encoder output states of shape (batch, src_len, hidden_size).

        Returns:
            Output tensor of shape (batch, tgt_len, hidden_size).
        """
        batch, tgt_len, _ = hidden_states.shape
        _, src_len, _ = encoder_hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(encoder_hidden_states).view(
            batch, src_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(encoder_hidden_states).view(
            batch, src_len, self.num_kv_heads, self.head_dim
        )

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_heads_per_group, dim=2)
        v = v.repeat_interleave(self.num_heads_per_group, dim=2)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Non-causal attention across encoder length dimension
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch, tgt_len, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out)


class GemmaT5DecoderLayer(nn.Module):
    """Transformer decoder block with causal self-attention, cross-attention, and MLP."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config)
        self.cross_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = GemmaCrossAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for decoder layer.

        Args:
            hidden_states: Decoder hidden states of shape (batch, tgt_len, hidden_size).
            encoder_hidden_states: Encoder output states of shape (batch, src_len, hidden_size).

        Returns:
            Output tensor of shape (batch, tgt_len, hidden_size).
        """
        # Causal Self-Attention
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(normed, is_causal=True)

        # Cross-Attention over encoder hidden states
        residual = hidden_states
        normed = self.cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.cross_attn(normed, encoder_hidden_states)

        # Feed-forward MLP
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normed)

        return hidden_states


class FoldGemmaT5(BaseFoldModel):
    """FoldGemmaT5 encoder-decoder model with cross-attention and autoregressive generation."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__(config)
        self.decoder_embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [GemmaT5DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def decode(
        self, decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Runs decoder layers and lm_head on decoder input tokens and encoder hidden states.

        Args:
            decoder_input_ids: Tensor of shape (batch, tgt_len).
            encoder_hidden_states: Tensor of shape (batch, src_len, hidden_size).

        Returns:
            Logits tensor of shape (batch, tgt_len, vocab_size).
        """
        embeds = self.decoder_embed_tokens(decoder_input_ids)
        decoder_states = embeds * math.sqrt(self.config.hidden_size)
        for layer in self.decoder_layers:
            decoder_states = layer(decoder_states, encoder_hidden_states)
        decoder_states = self.decoder_norm(decoder_states)
        return self.lm_head(decoder_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        plddt: torch.Tensor | None = None,
        plddt_threshold: float = 70.0,
    ) -> torch.Tensor:
        """Forward pass through encoder and decoder stacks.

        Args:
            input_ids: Encoder token IDs of shape (batch, src_len).
            decoder_input_ids: Decoder token IDs of shape (batch, tgt_len).
            plddt: Optional pLDDT score tensor of shape (batch, src_len).
            plddt_threshold: Confidence threshold for pLDDT mask.

        Returns:
            Logits tensor of shape (batch, tgt_len, vocab_size).
        """
        if decoder_input_ids is None:
            decoder_input_ids = input_ids

        encoder_hidden_states = self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
        return self.decode(decoder_input_ids, encoder_hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        plddt: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        bos_token_id: int = 2,
        eos_token_id: int | None = None,
        plddt_threshold: float = 70.0,
    ) -> torch.Tensor:
        """Autoregressively generates target tokens given encoder input IDs.

        Args:
            input_ids: Encoder input token IDs of shape (batch, src_len).
            plddt: Optional pLDDT confidence scores.
            max_new_tokens: Maximum number of target tokens to generate.
            bos_token_id: Beginning-of-sequence token ID (default 2).
            eos_token_id: End-of-sequence token ID.
            plddt_threshold: Threshold for pLDDT mask.

        Returns:
            Tensor of shape (batch, 1 + max_new_tokens) containing generated token sequences.
        """
        encoder_hidden_states = self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=input_ids.device
        )
        if eos_token_id is not None:
            finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            logits = self.decode(decoder_input_ids, encoder_hidden_states)
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return decoder_input_ids
