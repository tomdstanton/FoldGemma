"""Unit tests for Flax and PyTorch Gemma Bidirectional Encoder implementations."""

from typing import cast

import jax
from flax import nnx
import jax
import jax.numpy as jnp
from flax import nnx
import torch

from foldgemma.inference.models.gemma import FoldGemmaConfig as PyTorchFastProtT5Config
from foldgemma.inference.models.gemma import GemmaModel as PyTorchGemmaModel
from foldgemma.train.models.gemma import FoldGemmaConfig as FlaxFastProtT5Config
from foldgemma.train.models.gemma import GemmaModel as FlaxGemmaModel


def test_gemma_config_defaults() -> None:
    """Verify FastProtT5Config default hyperparameters for FastProtT5."""
    flax_config = FlaxFastProtT5Config()
    pytorch_config = PyTorchFastProtT5Config()

    for config in [flax_config, pytorch_config]:
        assert config.vocab_size == 64
        assert config.hidden_size == 256
        assert config.intermediate_size == 512
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4
        assert config.head_dim == 32
        assert config.rms_norm_eps == 1e-6
        assert config.rope_theta == 10000.0


def test_flax_gemma_model_init_and_forward() -> None:
    """Verify Flax Gemma model initialization and forward pass logits shape (batch, seq_len, 64)."""
    config = FlaxFastProtT5Config()
    model = FlaxGemmaModel(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 16
    dummy_input_ids = (
        jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape((batch_size, seq_len))
        % config.vocab_size
    )
    logits = cast(jax.Array, model(dummy_input_ids))

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not jnp.isnan(logits).any()
    assert not jnp.isinf(logits).any()


def test_pytorch_gemma_model_init_and_forward() -> None:
    """Verify PyTorch Gemma model initialization and forward pass logits shape."""
    config = PyTorchFastProtT5Config()
    model = PyTorchGemmaModel(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    dummy_input_ids = (
        torch.arange(batch_size * seq_len, dtype=torch.long).reshape((batch_size, seq_len))
        % config.vocab_size
    )

    with torch.no_grad():
        logits = model(dummy_input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_flax_bidirectional_attention_behavior() -> None:
    """Verify bidirectional attention in Flax (token 0 influenced by last token)."""
    config = FlaxFastProtT5Config(num_hidden_layers=1)
    model = FlaxGemmaModel(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(0)

    seq_1 = jnp.array([[5, 10, 15, 20, 25]], dtype=jnp.int32)
    # Modify only the last token position
    seq_2 = jnp.array([[5, 10, 15, 20, 30]], dtype=jnp.int32)

    logits_1 = cast(jax.Array, model(seq_1))
    logits_2 = cast(jax.Array, model(seq_2))

    # In bidirectional attention, token 0's output changes when token 4 is modified
    token_0_diff = jnp.max(jnp.abs(logits_1[0, 0, :] - logits_2[0, 0, :]))
    assert token_0_diff > 1e-5, f"Expected non-zero difference at token 0, got {token_0_diff}"


def test_pytorch_bidirectional_attention_behavior() -> None:
    """Verify bidirectional attention in PyTorch (token 0 influenced by last token)."""
    config = PyTorchFastProtT5Config(num_hidden_layers=1)
    model = PyTorchGemmaModel(config)
    model.eval()

    seq_1 = torch.tensor([[5, 10, 15, 20, 25]], dtype=torch.long)
    # Modify only the last token position
    seq_2 = torch.tensor([[5, 10, 15, 20, 30]], dtype=torch.long)

    with torch.no_grad():
        logits_1 = model(seq_1)
        logits_2 = model(seq_2)

    # In bidirectional attention, token 0's output changes when token 4 is modified
    token_0_diff = torch.max(torch.abs(logits_1[0, 0, :] - logits_2[0, 0, :])).item()
    assert token_0_diff > 1e-5, f"Expected non-zero difference at token 0, got {token_0_diff}"


def test_flax_base_fold_model_encode_and_plddt() -> None:
    """Verify Flax BaseFoldModel encode logic and pLDDT score mask ingestion."""
    from foldgemma.train.models.base import BaseFoldModel

    config = FlaxFastProtT5Config()
    model = BaseFoldModel(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size, seq_len = 2, 8
    dummy_input_ids = (
        jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape((batch_size, seq_len))
        % config.vocab_size
    )

    # Residues at indices (0, 1) and (1, 3) have pLDDT < 70.0
    plddt = jnp.full((batch_size, seq_len), 90.0, dtype=jnp.float32)
    plddt = plddt.at[0, 1].set(50.0)
    plddt = plddt.at[1, 3].set(65.0)
    encoded = cast(
        jax.Array,
        model.encode(dummy_input_ids, plddt=plddt, plddt_threshold=70.0),
    )

    assert encoded.shape == (batch_size, seq_len, config.hidden_size)
    assert not jnp.isnan(encoded).any()

    # Zero vector at masked positions
    assert jnp.all(encoded[0, 1, :] == 0.0)
    assert jnp.all(encoded[1, 3, :] == 0.0)
    # Non-zero vector at unmasked positions
    assert not jnp.all(encoded[0, 0, :] == 0.0)
    assert not jnp.all(encoded[1, 0, :] == 0.0)


def test_flax_foldgemma_forward() -> None:
    """Verify Flax FoldGemma forward pass and logit output shape."""
    from foldgemma.train.models.foldgemma import FoldGemma

    config = FlaxFastProtT5Config()
    model = FoldGemma(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size, seq_len = 2, 8
    dummy_input_ids = (
        jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape((batch_size, seq_len))
        % config.vocab_size
    )
    logits = cast(jax.Array, model(dummy_input_ids))

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not jnp.isnan(logits).any()
    assert not jnp.isinf(logits).any()


def test_flax_foldgemma_t5_forward_and_generate() -> None:
    """Verify Flax FoldGemmaT5 forward pass and autoregressive generate execution."""
    from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5

    config = FlaxFastProtT5Config()
    model = FoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size, enc_len, dec_len = 2, 8, 4
    input_ids = (
        jnp.arange(batch_size * enc_len, dtype=jnp.int32).reshape((batch_size, enc_len))
        % config.vocab_size
    )
    decoder_input_ids = (
        jnp.arange(batch_size * dec_len, dtype=jnp.int32).reshape((batch_size, dec_len))
        % config.vocab_size
    )
    plddt = jnp.full((batch_size, enc_len), 85.0, dtype=jnp.float32)

    # Test forward pass
    logits = cast(jax.Array, model(input_ids, decoder_input_ids, plddt=plddt))
    assert logits.shape == (batch_size, dec_len, config.vocab_size)
    assert not jnp.isnan(logits).any()

    # Test autoregressive generate()
    max_new_tokens = 6
    generated = model.generate(input_ids, plddt=plddt, max_new_tokens=max_new_tokens)
    assert generated.shape == (batch_size, 1 + max_new_tokens)
    assert not jnp.isnan(generated).any()


def test_pytorch_base_fold_model_encode_and_plddt() -> None:
    """Verify PyTorch BaseFoldModel encode logic and pLDDT score mask ingestion."""
    from foldgemma.inference.models.base import BaseFoldModel

    config = PyTorchFastProtT5Config()
    model = BaseFoldModel(config)
    model.eval()

    batch_size, seq_len = 2, 8
    dummy_input_ids = (
        torch.arange(batch_size * seq_len, dtype=torch.long).reshape((batch_size, seq_len))
        % config.vocab_size
    )

    # Residues at indices (0, 1) and (1, 3) have pLDDT < 70.0
    plddt = torch.full((batch_size, seq_len), 90.0, dtype=torch.float32)
    plddt[0, 1] = 50.0
    plddt[1, 3] = 65.0

    with torch.no_grad():
        encoded = model.encode(dummy_input_ids, plddt=plddt, plddt_threshold=70.0)

    assert encoded.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(encoded).any()
    assert not torch.isinf(encoded).any()

    # Zero vector at masked positions
    assert torch.all(encoded[0, 1, :] == 0.0)
    assert torch.all(encoded[1, 3, :] == 0.0)
    # Non-zero vector at unmasked positions
    assert not torch.all(encoded[0, 0, :] == 0.0)
    assert not torch.all(encoded[1, 0, :] == 0.0)


def test_pytorch_foldgemma_forward() -> None:
    """Verify PyTorch FoldGemma forward pass and logit output shape."""
    from foldgemma.inference.models.foldgemma import FoldGemma

    config = PyTorchFastProtT5Config()
    model = FoldGemma(config)
    model.eval()

    batch_size, seq_len = 2, 8
    dummy_input_ids = (
        torch.arange(batch_size * seq_len, dtype=torch.long).reshape((batch_size, seq_len))
        % config.vocab_size
    )

    with torch.no_grad():
        logits = model(dummy_input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_pytorch_foldgemma_t5_forward_and_generate() -> None:
    """Verify PyTorch FoldGemmaT5 forward pass and autoregressive generate execution."""
    from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5

    config = PyTorchFastProtT5Config()
    model = FoldGemmaT5(config)
    model.eval()

    batch_size, enc_len, dec_len = 2, 8, 4
    input_ids = (
        torch.arange(batch_size * enc_len, dtype=torch.long).reshape((batch_size, enc_len))
        % config.vocab_size
    )
    decoder_input_ids = (
        torch.arange(batch_size * dec_len, dtype=torch.long).reshape((batch_size, dec_len))
        % config.vocab_size
    )
    plddt = torch.full((batch_size, enc_len), 85.0, dtype=torch.float32)

    # Test forward pass
    with torch.no_grad():
        logits = model(input_ids, decoder_input_ids=decoder_input_ids, plddt=plddt)
    assert logits.shape == (batch_size, dec_len, config.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # Test autoregressive generate()
    max_new_tokens = 6
    generated = model.generate(input_ids, plddt=plddt, max_new_tokens=max_new_tokens)
    assert generated.shape == (batch_size, 1 + max_new_tokens)
    assert not torch.isnan(generated.float()).any()


def test_foldgemma_t5_eos_early_stopping() -> None:
    """Verify early stopping on eos_token_id for both Flax and PyTorch FoldGemmaT5."""
    from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as PyTorchFoldGemmaT5
    from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5 as FlaxFoldGemmaT5

    # Flax test
    flax_config = FlaxFastProtT5Config()
    flax_model = FlaxFoldGemmaT5(flax_config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(0)
    input_ids_flax = jnp.zeros((2, 4), dtype=jnp.int32)
    first_gen = flax_model.generate(input_ids_flax, max_new_tokens=1, eos_token_id=None)
    predicted_eos = int(first_gen[0, 1])
    gen_flax = flax_model.generate(input_ids_flax, max_new_tokens=10, eos_token_id=predicted_eos
    )
    assert gen_flax.shape[1] == 2

    # PyTorch test
    pytorch_config = PyTorchFastProtT5Config()
    pt_model = PyTorchFoldGemmaT5(pytorch_config)
    pt_model.eval()
    input_ids_pt = torch.zeros((2, 4), dtype=torch.long)
    first_gen_pt = pt_model.generate(input_ids_pt, max_new_tokens=1, eos_token_id=None)
    predicted_eos_pt = int(first_gen_pt[0, 1])
    gen_pt = pt_model.generate(
        input_ids_pt, max_new_tokens=10, eos_token_id=predicted_eos_pt
    )
    assert gen_pt.shape[1] == 2

