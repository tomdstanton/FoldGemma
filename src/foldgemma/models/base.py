"""PyTorch implementation of BaseFoldModel for FoldGemma."""

import math

import torch
import torch.nn as nn

from foldgemma.config import FoldGemmaConfig
from foldgemma.models.gemma import GemmaDecoderLayer, RMSNorm


class BaseFoldModel(nn.Module):
    """Abstract base class for Gemma transformer models with pLDDT mask ingestion."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def encode(
        self,
        input_ids: torch.Tensor,
        plddt: torch.Tensor | None = None,
        plddt_threshold: float = 70.0,
    ) -> torch.Tensor:
        """Encodes token sequence into hidden states with optional pLDDT masking.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            plddt: Optional pLDDT confidence score tensor of shape (batch, seq_len).
            plddt_threshold: Confidence threshold below which hidden representations are masked.

        Returns:
            Encoded hidden states tensor of shape (batch, seq_len, hidden_size).
        """
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if plddt is not None:
            mask = (plddt >= plddt_threshold).to(dtype=x.dtype)
            x = x * mask.unsqueeze(-1)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        plddt: torch.Tensor | None = None,
        plddt_threshold: float = 70.0,
    ) -> torch.Tensor:
        """Forward pass defaults to encode."""
        return self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
