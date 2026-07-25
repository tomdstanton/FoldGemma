"""Unit test bridge for JAX/Flax to PyTorch Gemma weight conversion and numerical equivalence."""

from typing import cast

import jax
from flax import nnx
import numpy as np
import torch

from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.export import FastProtT5Exporter, FoldGemmaExporter
from foldgemma.inference.models.foldgemma import FoldGemma as TorchFoldGemma
from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as TorchFoldGemmaT5
from foldgemma.inference.models.gemma import GemmaModel as PyTorchGemmaModel
from foldgemma.train.models.foldgemma import FoldGemma as FlaxFoldGemma
from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5 as FlaxFoldGemmaT5
from foldgemma.train.models.gemma import GemmaModel as FlaxGemmaModel


def test_bridge_flax_to_pytorch_equivalence() -> None:
    """Verify numerical equivalence between Flax GemmaModel and PyTorch GemmaModel."""
    config = FoldGemmaConfig()
    flax_model = FlaxGemmaModel(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(999)

    batch_size = 4
    seq_len = 32
    dummy_input_ids = jax.random.randint(
        key, (batch_size, seq_len), minval=0, maxval=config.vocab_size
    )
    flax_logits = cast(jax.Array, flax_model(dummy_input_ids))

    _, state = nnx.split(flax_model, nnx.Param)
    flax_params = state.to_pure_dict()
    exporter = FastProtT5Exporter(config)
    pytorch_state_dict = exporter.convert_weights(flax_params)

    pytorch_model = PyTorchGemmaModel(config)
    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()

    torch_input_ids = torch.from_numpy(np.array(dummy_input_ids, dtype=np.int64))
    with torch.no_grad():
        pytorch_logits = pytorch_model(torch_input_ids).numpy()

    max_diff = float(np.max(np.abs(np.array(flax_logits) - pytorch_logits)))
    assert max_diff < 1e-4, f"Flax vs PyTorch max logit difference {max_diff} >= 1e-4"


def test_bridge_foldgemma_flax_to_pytorch_equivalence() -> None:
    """Verify numerical equivalence between Flax FoldGemma and PyTorch FoldGemma."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
    flax_model = FlaxFoldGemma(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 16
    dummy_input_ids = jax.random.randint(
        key, (batch_size, seq_len), minval=0, maxval=config.vocab_size
    )
    flax_logits = cast(jax.Array, flax_model(dummy_input_ids))

    _, state = nnx.split(flax_model, nnx.Param)
    flax_params = state.to_pure_dict()
    exporter = FoldGemmaExporter(config)
    pytorch_state_dict = exporter.convert_weights(flax_params)

    pytorch_model = TorchFoldGemma(config)
    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()

    torch_input_ids = torch.from_numpy(np.array(dummy_input_ids, dtype=np.int64))
    with torch.no_grad():
        pytorch_logits = pytorch_model(torch_input_ids).numpy()

    max_diff = float(np.max(np.abs(np.array(flax_logits) - pytorch_logits)))
    assert max_diff < 1e-4, f"Flax vs PyTorch FoldGemma max logit difference {max_diff} >= 1e-4"


def test_bridge_foldgemma_t5_flax_to_pytorch_equivalence() -> None:
    """Verify numerical equivalence between Flax FoldGemmaT5 and PyTorch FoldGemmaT5."""
    config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA_T5)
    flax_model = FlaxFoldGemmaT5(config, rngs=nnx.Rngs(42))
    key = jax.random.PRNGKey(42)

    batch_size = 2
    enc_len = 16
    dec_len = 8
    dummy_input_ids = jax.random.randint(
        key, (batch_size, enc_len), minval=0, maxval=config.vocab_size
    )
    dummy_decoder_ids = jax.random.randint(
        key, (batch_size, dec_len), minval=0, maxval=config.vocab_size
    )
    flax_logits = cast(
        jax.Array, flax_model(dummy_input_ids, dummy_decoder_ids)
    )

    _, state = nnx.split(flax_model, nnx.Param)
    flax_params = state.to_pure_dict()
    exporter = FoldGemmaExporter(config)
    pytorch_state_dict = exporter.convert_weights(flax_params)

    pytorch_model = TorchFoldGemmaT5(config)
    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()

    torch_input_ids = torch.from_numpy(np.array(dummy_input_ids, dtype=np.int64))
    torch_decoder_ids = torch.from_numpy(np.array(dummy_decoder_ids, dtype=np.int64))
    with torch.no_grad():
        pytorch_logits = pytorch_model(
            torch_input_ids, decoder_input_ids=torch_decoder_ids
        ).numpy()

    max_diff = float(np.max(np.abs(np.array(flax_logits) - pytorch_logits)))
    assert max_diff < 1e-4, f"Flax vs PyTorch FoldGemmaT5 max logit difference {max_diff} >= 1e-4"
