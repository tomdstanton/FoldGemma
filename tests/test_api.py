"""Unit tests for FoldGemma API dynamic model instantiation and export bridge."""

import tempfile

import jax
from flax import nnx
import torch

from foldgemma.api import FoldGemmaInference, FoldGemmaTrainer
from foldgemma.config import ModelType
from foldgemma.inference.models.foldgemma import FoldGemma as TorchFoldGemma
from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as TorchFoldGemmaT5
from foldgemma.train.models.foldgemma import FoldGemma as FlaxFoldGemma
from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5 as FlaxFoldGemmaT5


def test_trainer_dynamic_instantiation_foldgemma() -> None:
    """Verify FoldGemmaTrainer instantiates FlaxFoldGemma for ModelType.FOLDGEMMA."""
    trainer = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA)
    trainer.initialize(0)
    assert isinstance(trainer.model, FlaxFoldGemma)
    assert trainer.config.model_type == ModelType.FOLDGEMMA
    assert trainer.model is not None


def test_trainer_dynamic_instantiation_foldgemma_t5() -> None:
    """Verify FoldGemmaTrainer instantiates FlaxFoldGemmaT5 for ModelType.FOLDGEMMA_T5."""
    trainer = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA_T5)
    trainer.initialize(0)
    assert isinstance(trainer.model, FlaxFoldGemmaT5)
    assert trainer.config.model_type == ModelType.FOLDGEMMA_T5
    assert trainer.model is not None


def test_trainer_string_model_type() -> None:
    """Verify FoldGemmaTrainer accepts string model_type argument."""
    trainer_fg = FoldGemmaTrainer(model_type="foldgemma")
    trainer_fg.initialize(0)
    assert isinstance(trainer_fg.model, FlaxFoldGemma)

    trainer_t5 = FoldGemmaTrainer(model_type="foldgemma_t5")
    trainer_t5.initialize(0)
    assert isinstance(trainer_t5.model, FlaxFoldGemmaT5)


def test_inference_dynamic_instantiation_foldgemma() -> None:
    """Verify FoldGemmaInference instantiates TorchFoldGemma for ModelType.FOLDGEMMA."""
    inference = FoldGemmaInference(model_type=ModelType.FOLDGEMMA, compile_model=False)
    assert isinstance(inference.model, TorchFoldGemma)
    assert inference.config.model_type == ModelType.FOLDGEMMA

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    logits = inference.predict(input_ids)
    assert logits.shape == (2, 4, inference.config.vocab_size)


def test_inference_dynamic_instantiation_foldgemma_t5() -> None:
    """Verify FoldGemmaInference instantiates TorchFoldGemmaT5 for ModelType.FOLDGEMMA_T5."""
    inference = FoldGemmaInference(model_type=ModelType.FOLDGEMMA_T5, compile_model=False)
    assert isinstance(inference.model, TorchFoldGemmaT5)
    assert inference.config.model_type == ModelType.FOLDGEMMA_T5

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    decoder_input_ids = torch.tensor([[2, 3], [2, 4]], dtype=torch.long)
    logits = inference.predict(input_ids, decoder_input_ids=decoder_input_ids)
    assert logits.shape == (2, 2, inference.config.vocab_size)

    generated = inference.generate(input_ids, max_new_tokens=5)
    assert generated.shape == (2, 6)


def test_export_to_pytorch_foldgemma_and_t5() -> None:
    """Verify end-to-end export from Trainer to PyTorch inference weights."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test FoldGemma export
        trainer_fg = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA)
        trainer_fg.initialize(123)
        fg_ckpt_path = f"{tmp_dir}/foldgemma.safetensors"
        torch_fg = trainer_fg.export_to_pytorch(fg_ckpt_path)
        assert isinstance(torch_fg, TorchFoldGemma)

        # Test FoldGemmaT5 export
        trainer_t5 = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA_T5)
        trainer_t5.initialize(456)
        t5_ckpt_path = f"{tmp_dir}/foldgemma_t5.safetensors"
        torch_t5 = trainer_t5.export_to_pytorch(t5_ckpt_path)
        assert isinstance(torch_t5, TorchFoldGemmaT5)
