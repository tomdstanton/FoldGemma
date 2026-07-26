"""PyTorch implementation of FoldGemma classification model."""

import torch
import torch.nn as nn

from foldgemma.config import FoldGemmaConfig
from foldgemma.models.base import BaseFoldModel


class FoldGemma(BaseFoldModel):
    """FoldGemma model with linear classification head returning 3di predictions."""

    def __init__(self, config: FoldGemmaConfig) -> None:
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        plddt: torch.Tensor | None = None,
        plddt_threshold: float = 70.0,
    ) -> torch.Tensor:
        """Forward pass computing 3di logits.

        Args:
            input_ids: Tensor of shape (batch, seq_len).
            plddt: Optional tensor of shape (batch, seq_len).
            plddt_threshold: Confidence threshold for pLDDT mask.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        hidden_states = self.encode(input_ids, plddt=plddt, plddt_threshold=plddt_threshold)
        return self.lm_head(hidden_states)
