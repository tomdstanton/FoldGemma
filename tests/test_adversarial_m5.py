"""Adversarial stress test suite for FoldGemma.

Tests model instantiations (FoldGemma, FoldT5Gemma) and high-level API
wrappers (FoldGemmaTrainer) under edge-cases, extreme bounds, and dynamic
configurations.
"""

import torch

from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.models.fold_t5gemma import FoldT5Gemma
from foldgemma.models.foldgemma import FoldGemma
from foldgemma.trainer import FoldGemmaTrainer

# ============================================================================
# Section 1: Direct Model Instantiations & Corner Cases (Batch Size, Seq Len, pLDDT)
# ============================================================================


def test_trainer_dynamic_instantiation() -> None:
    """Test FoldGemmaTrainer dynamic instantiation via Enum, string, and config override."""
    config = FoldGemmaConfig(model_type=ModelType.GEMMA)

    # Case A: Instantiation with string "t5gemma" overriding config
    trainer_t5 = FoldGemmaTrainer(config, model_type="t5gemma")
    trainer_t5.initialize(0)
    assert isinstance(trainer_t5.model, FoldT5Gemma)
    assert trainer_t5.config.model_type == ModelType.T5GEMMA

    # Case B: Instantiation with ModelType enum
    trainer_fg = FoldGemmaTrainer(config, model_type=ModelType.GEMMA)
    trainer_fg.initialize(0)
    assert isinstance(trainer_fg.model, FoldGemma)
    assert trainer_fg.config.model_type == ModelType.GEMMA

    # Case C: Instantiation with default config model_type
    t5_config = FoldGemmaConfig(model_type=ModelType.T5GEMMA)
    trainer_default_t5 = FoldGemmaTrainer(t5_config)
    trainer_default_t5.initialize(0)
    assert isinstance(trainer_default_t5.model, FoldT5Gemma)


def test_all_masked_out_plddt_propagation() -> None:
    """Test that when pLDDT is all < threshold, model produces clean valid representations."""
    config = FoldGemmaConfig(model_type=ModelType.GEMMA)

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
    assert (encoded_torch == 0.0).all(), "PyTorch encoded representations with 100% masked pLDDT are not all zeros"
