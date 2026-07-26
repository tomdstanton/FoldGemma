"""Unit tests for Flax and PyTorch Gemma Bidirectional Encoder implementations."""

from typing import cast

import torch

from foldgemma.models.gemma import FoldGemmaConfig
from foldgemma.models.gemma import GemmaModel


def test_gemma_config_defaults() -> None:
    """Verify FastProtT5Config default hyperparameters for FastProtT5."""
    config = FoldGemmaConfig()

    assert config.vocab_size == 64
    assert config.hidden_size == 256
    assert config.intermediate_size == 512
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4
    assert config.head_dim == 32
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10000.0




def test_gemma_model_init_and_forward() -> None:
    """Verify PyTorch Gemma model initialization and forward pass logits shape."""
    config = FoldGemmaConfig()
    model = GemmaModel(config)
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




def test_bidirectional_attention_behavior() -> None:
    """Verify bidirectional attention in PyTorch (token 0 influenced by last token)."""
    config = FoldGemmaConfig(num_hidden_layers=1)
    model = GemmaModel(config)
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








def test_base_fold_model_encode_and_plddt() -> None:
    """Verify PyTorch BaseFoldModel encode logic and pLDDT score mask ingestion."""
    from foldgemma.models.base import BaseFoldModel

    config = FoldGemmaConfig()
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


def test_foldgemma_forward() -> None:
    """Verify PyTorch FoldGemma forward pass and logit output shape."""
    from foldgemma.models.foldgemma import FoldGemma

    config = FoldGemmaConfig()
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


def test_foldgemma_t5_forward_and_generate() -> None:
    """Verify PyTorch FoldGemmaT5 forward pass and autoregressive generate execution."""
    from foldgemma.models.foldgemma_t5 import FoldGemmaT5

    config = FoldGemmaConfig()
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
    """Verify early stopping on eos_token_id for PyTorch FoldGemmaT5."""
    from foldgemma.models.foldgemma_t5 import FoldGemmaT5

    pytorch_config = FoldGemmaConfig()
    pt_model = FoldGemmaT5(pytorch_config)
    pt_model.eval()
    input_ids_pt = torch.zeros((2, 4), dtype=torch.long)
    first_gen_pt = pt_model.generate(input_ids_pt, max_new_tokens=1, eos_token_id=None)
    predicted_eos_pt = int(first_gen_pt[0, 1])
    gen_pt = pt_model.generate(
        input_ids_pt, max_new_tokens=10, eos_token_id=predicted_eos_pt
    )
    assert gen_pt.shape[1] == 2

