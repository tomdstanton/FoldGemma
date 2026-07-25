"""Adversarial stress test suite for FoldGemma refactor (m5).

Tests all 4 model instantiations (Flax FoldGemma, Flax FoldGemmaT5, PyTorch
FoldGemma, PyTorch FoldGemmaT5) and high-level API wrappers
(FoldGemmaTrainer, FoldGemmaInference) under edge-cases, extreme bounds,
custom generation parameters, and dynamic configurations.
"""


from typing import cast

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

# ============================================================================
# Section 1: Direct Model Instantiations & Corner Cases (Batch Size, Seq Len, pLDDT)
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [1, 64, 128])
@pytest.mark.parametrize("plddt_case", ["none", "zero", "hundred", "all_masked", "none_masked"])
def test_flax_foldgemma_corner_cases(batch_size: int, seq_len: int, plddt_case: str) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    model = FlaxFoldGemma(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    dummy_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    if plddt_case == "none":
        dummy_plddt = None
    elif plddt_case == "zero":
        dummy_plddt = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
    elif plddt_case == "hundred":
        dummy_plddt = jnp.full((batch_size, seq_len), 100.0, dtype=jnp.float32)
    elif plddt_case == "all_masked":
        dummy_plddt = jnp.full((batch_size, seq_len), 50.0, dtype=jnp.float32)  # < threshold 70.0
    elif plddt_case == "none_masked":
        dummy_plddt = jnp.full((batch_size, seq_len), 85.0, dtype=jnp.float32)  # >= threshold 70.0
    else:
        raise ValueError(f"Unknown plddt_case: {plddt_case}")
    logits = cast(
        jax.Array,
        model(dummy_input_ids, plddt=dummy_plddt, plddt_threshold=70.0),
    )

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Flax FoldGemma shape mismatch: {logits.shape} vs {expected_shape}"
    )
    assert not jnp.isnan(logits).any(), "Flax FoldGemma logits contain NaN"
    assert not jnp.isinf(logits).any(), "Flax FoldGemma logits contain Inf"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("enc_len,dec_len", [(1, 1), (64, 16), (128, 64)])
@pytest.mark.parametrize("plddt_case", ["none", "zero", "hundred", "all_masked", "none_masked"])
def test_flax_foldgemma_t5_corner_cases(
    batch_size: int, enc_len: int, dec_len: int, plddt_case: str
) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = FlaxFoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    input_ids = jnp.ones((batch_size, enc_len), dtype=jnp.int32)
    decoder_input_ids = jnp.ones((batch_size, dec_len), dtype=jnp.int32)

    if plddt_case == "none":
        dummy_plddt = None
    elif plddt_case == "zero":
        dummy_plddt = jnp.zeros((batch_size, enc_len), dtype=jnp.float32)
    elif plddt_case == "hundred":
        dummy_plddt = jnp.full((batch_size, enc_len), 100.0, dtype=jnp.float32)
    elif plddt_case == "all_masked":
        dummy_plddt = jnp.full((batch_size, enc_len), 30.0, dtype=jnp.float32)
    elif plddt_case == "none_masked":
        dummy_plddt = jnp.full((batch_size, enc_len), 90.0, dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown plddt_case: {plddt_case}")
    logits = cast(
        jax.Array,
        model(input_ids, decoder_input_ids, plddt=dummy_plddt, plddt_threshold=70.0),
    )

    expected_logits_shape = (batch_size, dec_len, config.vocab_size)
    assert logits.shape == expected_logits_shape, (
        f"Flax FoldGemmaT5 logits shape mismatch: {logits.shape}"
    )
    assert not jnp.isnan(logits).any(), "Flax FoldGemmaT5 logits contain NaN"
    assert not jnp.isinf(logits).any(), "Flax FoldGemmaT5 logits contain Inf"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [1, 64, 128])
@pytest.mark.parametrize("plddt_case", ["none", "zero", "hundred", "all_masked", "none_masked"])
def test_torch_foldgemma_corner_cases(batch_size: int, seq_len: int, plddt_case: str) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    model = TorchFoldGemma(config)
    model.eval()

    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)

    if plddt_case == "none":
        dummy_plddt = None
    elif plddt_case == "zero":
        dummy_plddt = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    elif plddt_case == "hundred":
        dummy_plddt = torch.full((batch_size, seq_len), 100.0, dtype=torch.float32)
    elif plddt_case == "all_masked":
        dummy_plddt = torch.full((batch_size, seq_len), 50.0, dtype=torch.float32)
    elif plddt_case == "none_masked":
        dummy_plddt = torch.full((batch_size, seq_len), 85.0, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown plddt_case: {plddt_case}")

    with torch.no_grad():
        logits = model(input_ids, plddt=dummy_plddt, plddt_threshold=70.0)

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Torch FoldGemma shape mismatch: {logits.shape}"
    assert not torch.isnan(logits).any(), "Torch FoldGemma logits contain NaN"
    assert not torch.isinf(logits).any(), "Torch FoldGemma logits contain Inf"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("enc_len,dec_len", [(1, 1), (64, 16), (128, 64)])
@pytest.mark.parametrize("plddt_case", ["none", "zero", "hundred", "all_masked", "none_masked"])
def test_torch_foldgemma_t5_corner_cases(
    batch_size: int, enc_len: int, dec_len: int, plddt_case: str
) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = TorchFoldGemmaT5(config)
    model.eval()

    input_ids = torch.ones((batch_size, enc_len), dtype=torch.long)
    decoder_input_ids = torch.ones((batch_size, dec_len), dtype=torch.long)

    if plddt_case == "none":
        dummy_plddt = None
    elif plddt_case == "zero":
        dummy_plddt = torch.zeros((batch_size, enc_len), dtype=torch.float32)
    elif plddt_case == "hundred":
        dummy_plddt = torch.full((batch_size, enc_len), 100.0, dtype=torch.float32)
    elif plddt_case == "all_masked":
        dummy_plddt = torch.full((batch_size, enc_len), 30.0, dtype=torch.float32)
    elif plddt_case == "none_masked":
        dummy_plddt = torch.full((batch_size, enc_len), 90.0, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown plddt_case: {plddt_case}")

    with torch.no_grad():
        logits = model(
            input_ids, decoder_input_ids=decoder_input_ids, plddt=dummy_plddt, plddt_threshold=70.0
        )

    expected_logits_shape = (batch_size, dec_len, config.vocab_size)
    assert logits.shape == expected_logits_shape, (
        f"Torch FoldGemmaT5 shape mismatch: {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "Torch FoldGemmaT5 logits contain NaN"
    assert not torch.isinf(logits).any(), "Torch FoldGemmaT5 logits contain Inf"


# ============================================================================
# Section 2: Generation with Custom Parameters (bos_token_id, eos_token_id, max_new_tokens)
# ============================================================================

@pytest.mark.parametrize("bos_token_id", [0, 2, 10])
@pytest.mark.parametrize("eos_token_id", [1, 5, 20])
@pytest.mark.parametrize("max_new_tokens", [1, 8, 16])
def test_flax_generate_custom_tokens(
    bos_token_id: int, eos_token_id: int, max_new_tokens: int
) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = FlaxFoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size, enc_len = 2, 16
    input_ids = jnp.ones((batch_size, enc_len), dtype=jnp.int32)
    dummy_decoder_input = jnp.ones((batch_size, 4), dtype=jnp.int32)

    generated = model.generate(input_ids,
        max_new_tokens=max_new_tokens,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    # First token must be bos_token_id
    assert (generated[:, 0] == bos_token_id).all(), (
        f"Expected BOS token {bos_token_id}, got {generated[:, 0]}"
    )
    assert generated.shape == (batch_size, 1 + max_new_tokens), (
        f"Generated shape mismatch: {generated.shape}"
    )


@pytest.mark.parametrize("bos_token_id", [0, 2, 10])
@pytest.mark.parametrize("eos_token_id", [1, 5, 20])
@pytest.mark.parametrize("max_new_tokens", [1, 8, 16])
def test_torch_generate_custom_tokens(
    bos_token_id: int, eos_token_id: int, max_new_tokens: int
) -> None:
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    model = TorchFoldGemmaT5(config)
    model.eval()

    batch_size, enc_len = 2, 16
    input_ids = torch.ones((batch_size, enc_len), dtype=torch.long)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    # First token must be bos_token_id
    assert (generated[:, 0] == bos_token_id).all(), (
        f"Expected BOS token {bos_token_id}, got {generated[:, 0]}"
    )
    # Unless early stopped, shape is (batch_size, 1 + max_new_tokens)
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= 1 + max_new_tokens


def test_flax_vs_torch_eos_token_early_stopping_discrepancy() -> None:
    """Stress test: inspect EOS token early stopping behavior difference between Flax and Torch."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    
    # In Flax FoldGemmaT5.generate, eos_token_id is passed as parameter but unused in loop.
    # We verify that Flax generate always returns (batch_size, 1 + max_new_tokens).
    flax_model = FlaxFoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)
    input_ids_flax = jnp.ones((2, 8), dtype=jnp.int32)
    
    gen_flax = flax_model.generate(input_ids_flax, max_new_tokens=10, eos_token_id=1)
    assert gen_flax.shape == (2, 11), f"Flax generated shape {gen_flax.shape} mismatch"

    # In PyTorch FoldGemmaT5.generate, early stopping checks if (next_tokens == eos_token_id).all().
    # If all tokens in a batch match eos_token_id, PyTorch breaks early.
    torch_model = TorchFoldGemmaT5(config)
    torch_model.eval()
    input_ids_torch = torch.ones((2, 8), dtype=torch.long)
    with torch.no_grad():
        gen_torch = torch_model.generate(input_ids_torch, max_new_tokens=10, eos_token_id=1)
    assert gen_torch.shape[0] == 2


# ============================================================================
# Section 3: Dynamic Instantiation & Edge Cases in API Wrappers
# ============================================================================

def test_trainer_dynamic_instantiation() -> None:
    """Test FoldGemmaTrainer dynamic instantiation via Enum, string, and config override."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)

    # Case A: Instantiation with string "foldgemma_t5" overriding config
    trainer_t5 = FoldGemmaTrainer(config, model_type="foldgemma_t5")
    trainer_t5.initialize(0)
    assert isinstance(trainer_t5.model, FlaxFoldGemmaT5)
    assert trainer_t5.config.model_type == ModelType.FOLDGEMMA_T5

    # Case B: Instantiation with ModelType enum
    trainer_fg = FoldGemmaTrainer(config, model_type=ModelType.FOLDGEMMA)
    trainer_fg.initialize(0)
    assert isinstance(trainer_fg.model, FlaxFoldGemma)
    assert trainer_fg.config.model_type == ModelType.FOLDGEMMA

    # Case C: Instantiation with default config model_type
    t5_config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    trainer_default_t5 = FoldGemmaTrainer(t5_config)
    trainer_default_t5.initialize(0)
    assert isinstance(trainer_default_t5.model, FlaxFoldGemmaT5)


def test_inference_dynamic_instantiation() -> None:
    """Test FoldGemmaInference dynamic instantiation via Enum, string, and config override."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)

    # Case A: Instantiation with string "foldgemma_t5" overriding config
    inf_t5 = FoldGemmaInference(config, compile_model=False, model_type="foldgemma_t5")
    assert isinstance(inf_t5.model, TorchFoldGemmaT5)
    assert inf_t5.config.model_type == ModelType.FOLDGEMMA_T5

    # Case B: Instantiation with ModelType enum
    inf_fg = FoldGemmaInference(config, compile_model=False, model_type=ModelType.FOLDGEMMA)
    assert isinstance(inf_fg.model, TorchFoldGemma)
    assert inf_fg.config.model_type == ModelType.FOLDGEMMA

    # Case C: Instantiation with default config model_type
    t5_config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    inf_default_t5 = FoldGemmaInference(t5_config, compile_model=False)
    assert isinstance(inf_default_t5.model, TorchFoldGemmaT5)


def test_non_t5_generation_exception() -> None:
    """Test calling generate on non-T5 models raises appropriate AttributeError."""
    # PyTorch FoldGemmaInference with FOLDGEMMA
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    inf = FoldGemmaInference(config, compile_model=False)
    input_ids = torch.ones((2, 8), dtype=torch.long)
    
    with pytest.raises(AttributeError, match="does not support generation"):
        inf.generate(input_ids)

    # Flax FoldGemmaTrainer with FOLDGEMMA
    trainer = FoldGemmaTrainer(config)
    trainer.initialize(42)
    assert trainer.model is not None
    with pytest.raises(AttributeError):
        trainer.model.generate(jnp.ones((2, 8), dtype=jnp.int32))


def test_inference_dtype_autocast_and_plddt_handling() -> None:
    """Test PyTorch FoldGemmaInference predict and generate with float32 plddt tensors."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    inf = FoldGemmaInference(config, compile_model=False)

    batch_size, enc_len, dec_len = 4, 32, 16
    input_ids = torch.ones((batch_size, enc_len), dtype=torch.long)
    decoder_input_ids = torch.ones((batch_size, dec_len), dtype=torch.long)
    plddt = torch.full((batch_size, enc_len), 80.0, dtype=torch.float32)

    # Predict call
    logits = inf.predict(input_ids, decoder_input_ids=decoder_input_ids, plddt=plddt)
    assert logits.shape == (batch_size, dec_len, config.vocab_size)
    assert logits.dtype == torch.bfloat16

    # Generate call with plddt
    generated = inf.generate(input_ids, plddt=plddt, max_new_tokens=8)
    assert generated.shape == (batch_size, 1 + 8)


def test_all_masked_out_plddt_propagation() -> None:
    """Test that when pLDDT is all < threshold, model produces clean valid representations."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    
    # Flax
    flax_model = FlaxFoldGemma(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)
    dummy_input_ids = jnp.ones((2, 16), dtype=jnp.int32)
    dummy_plddt_flax = jnp.full((2, 16), 10.0, dtype=jnp.float32)  # threshold 70.0
    
    # Hidden states after encode should be all zeros
    encoded_flax = cast(
        jax.Array,
        flax_model.encode(
            dummy_input_ids,
            plddt=dummy_plddt_flax,
            plddt_threshold=70.0,
        ),
    )
    assert (encoded_flax == 0.0).all(), (
        "Flax encoded representations with 100% masked pLDDT are not all zeros"
    )

    # PyTorch
    torch_model = TorchFoldGemma(config)
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
