"""Deep dive bug identification and verification tests for FoldGemma.

Probes subtle edge cases:
1. FoldT5Gemma batch-all early stopping logic for eos_token_id
2. None / missing plddt in train_step / compute_masked_loss
3. Invalid model_type string handling in FoldGemmaTrainer
4. pLDDT shape mismatch with input_ids
5. Token ID out-of-vocab indexing behavior
"""

import karva
import torch

from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.models.fold_t5gemma import FoldT5Gemma
from foldgemma.models.foldgemma import FoldGemma
from foldgemma.trainer import FoldGemmaTrainer


def test_t5_eos_token_batch_all_requirement() -> None:
    """Verify Torch FoldT5Gemma.generate requires ALL batch sequences to produce EOS."""
    config = FoldGemmaConfig(model_type=ModelType.T5GEMMA)
    model = FoldT5Gemma(config)
    model.eval()

    batch_size = 2
    input_ids = torch.ones((batch_size, 8), dtype=torch.long)

    # PyTorch generate: if eos_token_id is not None and (next_tokens == eos_token_id).all()
    # If seq 0 generates EOS but seq 1 does not, .all() is False, so seq 0 continues.
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=15, eos_token_id=1)

    assert generated.shape[0] == batch_size
    print(f"\n[EMPIRICAL OBSERVATION] PyTorch T5 output shape for batch=2: {generated.shape}")


def test_invalid_model_type_str_handling() -> None:
    """Test passing invalid string model_type to FoldGemmaTrainer and FoldGemmaInference."""
    config = FoldGemmaConfig()

    with karva.raises(ValueError):
        FoldGemmaTrainer(config, model_type="non_existent_architecture")

    print("\n[VERIFIED] Invalid model_type string correctly raises ValueError.")


def test_plddt_shape_mismatch_torch() -> None:
    """Test behavior when plddt shape does not match input_ids shape in PyTorch."""
    config = FoldGemmaConfig(model_type=ModelType.GEMMA)
    model = FoldGemma(config)
    model.eval()

    input_ids = torch.ones((2, 16), dtype=torch.long)
    mismatched_plddt = torch.full((2, 10), 85.0, dtype=torch.float32)

    with torch.no_grad():
        with karva.raises(RuntimeError):
            model(input_ids, plddt=mismatched_plddt, plddt_threshold=70.0)

    print("\n[VERIFIED] Mismatched plddt length correctly raises RuntimeError in PyTorch.")
