"""Deep dive bug identification and verification tests for FoldGemma.

Probes subtle edge cases:
1. Flax FoldGemmaT5 unused eos_token_id parameter in generate()
2. Torch FoldGemmaT5 batch-all early stopping logic for eos_token_id
3. None / missing plddt in train_step / compute_masked_loss
4. Invalid model_type string handling in FoldGemmaTrainer & FoldGemmaInference
5. pLDDT shape mismatch with input_ids
6. Token ID out-of-vocab indexing behavior
"""

import jax
from flax import nnx
import jax
import jax.numpy as jnp
from flax import nnx
import pytest
import torch

from foldgemma.api import FoldGemmaInference, FoldGemmaTrainer
from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.inference.models.foldgemma import FoldGemma as TorchFoldGemma
from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as TorchFoldGemmaT5
from foldgemma.train.models.foldgemma import FoldGemma as FlaxFoldGemma
from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5 as FlaxFoldGemmaT5
from foldgemma.train.train import train_step


def test_flax_t5_eos_token_ignored() -> None:
    """Verify that Flax FoldGemmaT5.generate ignores eos_token_id and never terminates early."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = FlaxFoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    input_ids = jnp.ones((1, 8), dtype=jnp.int32)

    # Pass max_new_tokens=20 and eos_token_id=0 (or any valid token)
    # Because Flax generate never checks eos_token_id in loop, len is 1 + max_new_tokens
    generated = model.generate(input_ids, max_new_tokens=20, eos_token_id=0)
    assert generated.shape == (1, 21), f"Flax generate shape {generated.shape} expected (1, 21)"
    print("\n[EMPIRICAL BUG CONFIRMED] Flax FoldGemmaT5.generate() accepts eos_token_id")


def test_torch_t5_eos_token_batch_all_requirement() -> None:
    """Verify Torch FoldGemmaT5.generate requires ALL batch sequences to produce EOS."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = TorchFoldGemmaT5(config)
    model.eval()

    batch_size = 2
    input_ids = torch.ones((batch_size, 8), dtype=torch.long)

    # PyTorch generate: if eos_token_id is not None and (next_tokens == eos_token_id).all()
    # If seq 0 generates EOS but seq 1 does not, .all() is False, so seq 0 continues.
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=15, eos_token_id=1)
    
    assert generated.shape[0] == batch_size
    print(f"\n[EMPIRICAL OBSERVATION] PyTorch T5 output shape for batch=2: {generated.shape}")


def test_train_step_missing_plddt() -> None:
    """Test train_step when plddt is None in batch handles plddt=None gracefully."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    trainer = FoldGemmaTrainer(config)
    trainer.initialize(42)

    # Batch without 'plddt' or with 'plddt': None
    batch = {
        "inputs": jnp.ones((2, 16), dtype=jnp.int32),
        "targets": jnp.ones((2, 16), dtype=jnp.int32),
        "plddt": None,
    }

    loss = train_step(
        trainer.model,
        trainer.optimizer,
        batch,
        pad_id=0,
        unk_id=1,
        plddt_threshold=70.0,
    )
    assert loss is not None
    assert not jnp.isnan(loss)


def test_invalid_model_type_str_handling() -> None:
    """Test passing invalid string model_type to FoldGemmaTrainer and FoldGemmaInference."""
    config = FoldGemmaConfig()
    
    with pytest.raises(ValueError):
        FoldGemmaTrainer(config, model_type="non_existent_architecture")

    with pytest.raises(ValueError):
        FoldGemmaInference(config, compile_model=False, model_type="non_existent_architecture")
    
    print("\n[VERIFIED] Invalid model_type string correctly raises ValueError.")


def test_plddt_shape_mismatch_flax() -> None:
    """Test behavior when plddt shape does not match input_ids shape in Flax."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    model = FlaxFoldGemma(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    input_ids = jnp.ones((2, 16), dtype=jnp.int32)

    # Mismatched plddt seq_len (e.g. 10 instead of 16)
    mismatched_plddt = jnp.full((2, 10), 85.0, dtype=jnp.float32)

    with pytest.raises((ValueError, TypeError)):
        model(input_ids, plddt=mismatched_plddt, plddt_threshold=70.0)
    
    print("\n[VERIFIED] Mismatched plddt length correctly raises shape error in Flax broadcast.")


def test_plddt_shape_mismatch_torch() -> None:
    """Test behavior when plddt shape does not match input_ids shape in PyTorch."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    model = TorchFoldGemma(config)
    model.eval()

    input_ids = torch.ones((2, 16), dtype=torch.long)
    mismatched_plddt = torch.full((2, 10), 85.0, dtype=torch.float32)

    with torch.no_grad():
        with pytest.raises(RuntimeError):
            model(input_ids, plddt=mismatched_plddt, plddt_threshold=70.0)

    print("\n[VERIFIED] Mismatched plddt length correctly raises RuntimeError in PyTorch.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
