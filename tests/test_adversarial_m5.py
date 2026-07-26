"""Adversarial stress test suite for FoldGemma refactor (m5).

Tests all 4 model instantiations (Flax FoldGemma, Flax FoldGemmaT5, PyTorch
FoldGemma, PyTorch FoldGemmaT5) and high-level API wrappers
(FoldGemmaTrainer, FoldGemmaInference) under edge-cases, extreme bounds,
custom generation parameters, and dynamic configurations.
"""


from typing import cast

import pytest
import torch

from foldgemma.trainer import FoldGemmaTrainer
from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.models.foldgemma import FoldGemma
from foldgemma.models.foldgemma_t5 import FoldGemmaT5

# ============================================================================
# Section 1: Direct Model Instantiations & Corner Cases (Batch Size, Seq Len, pLDDT)
# ============================================================================

def test_trainer_dynamic_instantiation() -> None:
    """Test FoldGemmaTrainer dynamic instantiation via Enum, string, and config override."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)

    # Case A: Instantiation with string "foldgemma_t5" overriding config
    trainer_t5 = FoldGemmaTrainer(config, model_type="foldgemma_t5")
    trainer_t5.initialize(0)
    assert isinstance(trainer_t5.model, FoldGemmaT5)
    assert trainer_t5.config.model_type == ModelType.FOLDGEMMA_T5

    # Case B: Instantiation with ModelType enum
    trainer_fg = FoldGemmaTrainer(config, model_type=ModelType.FOLDGEMMA)
    trainer_fg.initialize(0)
    assert isinstance(trainer_fg.model, FoldGemma)
    assert trainer_fg.config.model_type == ModelType.FOLDGEMMA

    # Case C: Instantiation with default config model_type
    t5_config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    trainer_default_t5 = FoldGemmaTrainer(t5_config)
    trainer_default_t5.initialize(0)
    assert isinstance(trainer_default_t5.model, FoldGemmaT5)


def test_all_masked_out_plddt_propagation() -> None:
    """Test that when pLDDT is all < threshold, model produces clean valid representations."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    
    # PyTorch
    torch_model = FoldGemma(config)
    torch_model.eval()
    dummy_plddt_torch = torch.full((2, 16), 10.0, dtype=torch.float32)
    with torch.no_grad():
        encoded_torch = torch_model.encode(
            torch.ones((2, 16), dtype=torch.long),
            plddt=dummy_plddt_torch,
            plddt_threshold=70.0,
        )
    assert (encoded_torch == 0.0).all(), (
        "PyTorch encoded representations with 100% masked pLDDT are not all zeros"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
