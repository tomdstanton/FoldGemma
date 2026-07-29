"""Composite loss function with quality and token masking for FoldGemma in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldgemma.data.vocabulary import PAD_ID, UNK_ID

class MaskedCrossEntropyLoss(nn.Module):
    """Composite masked cross entropy loss using PyTorch."""
    
    def __init__(self, pad_id: int = PAD_ID, unk_id: int = UNK_ID, plddt_threshold: float = 70.0):
        super().__init__()
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.plddt_threshold = plddt_threshold

    def compute_mask(self, targets: torch.Tensor, plddt: torch.Tensor | None = None) -> torch.Tensor:
        """Construct composite binary mask: (targets != pad_id) & (targets != unk_id) & (plddt >= threshold)."""
        valid_target = (targets != self.pad_id) & (targets != self.unk_id)
        if plddt is not None:
            return valid_target & (plddt >= self.plddt_threshold)
        return valid_target

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        plddt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute composite masked cross entropy loss.
        
        Returns:
            Scalar average loss divided ONLY by sum of valid mask.
        """
        vocab_size = logits.size(-1)
        raw_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1).long(), reduction='none')
        raw_loss = raw_loss.view(targets.shape)
        
        mask = self.compute_mask(targets, plddt).to(logits.dtype)
        
        masked_loss = raw_loss * mask
        valid_count = mask.sum()
        return masked_loss.sum() / valid_count.clamp(min=1.0)
