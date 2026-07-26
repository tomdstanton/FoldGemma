"""Unit tests for FoldGemma API dynamic model instantiation and export bridge."""

import tempfile
import torch

from foldgemma.trainer import FoldGemmaTrainer
from foldgemma.config import ModelType
from foldgemma.models.foldgemma import FoldGemma
from foldgemma.models.foldgemma_t5 import FoldGemmaT5

def test_trainer_dynamic_instantiation_foldgemma() -> None:
    """Verify FoldGemmaTrainer instantiates FoldGemma for ModelType.FOLDGEMMA."""
    trainer = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA)
    trainer.initialize(0)
    assert isinstance(trainer.model, FoldGemma)
    assert trainer.config.model_type == ModelType.FOLDGEMMA
    assert trainer.model is not None

def test_trainer_dynamic_instantiation_foldgemma_t5() -> None:
    """Verify FoldGemmaTrainer instantiates FoldGemmaT5 for ModelType.FOLDGEMMA_T5."""
    trainer = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA_T5)
    trainer.initialize(0)
    assert isinstance(trainer.model, FoldGemmaT5)
    assert trainer.config.model_type == ModelType.FOLDGEMMA_T5
    assert trainer.model is not None

def test_trainer_string_model_type() -> None:
    """Verify FoldGemmaTrainer accepts string model_type argument."""
    trainer_fg = FoldGemmaTrainer(model_type="foldgemma")
    trainer_fg.initialize(0)
    assert isinstance(trainer_fg.model, FoldGemma)

    trainer_t5 = FoldGemmaTrainer(model_type="foldgemma_t5")
    trainer_t5.initialize(0)
    assert isinstance(trainer_t5.model, FoldGemmaT5)

def test_trainer_save_and_load_checkpoint() -> None:
    """Verify end-to-end save and load checkpoint from Trainer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test FoldGemma checkpointing
        trainer_fg = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA)
        trainer_fg.initialize(123)
        trainer_fg.save_checkpoint(tmp_dir)
        
        # Modify weights slightly
        with torch.no_grad():
            for param in trainer_fg.model.parameters():
                param.add_(1.0)
                
        trainer_fg.load_checkpoint(tmp_dir)
        
        # Test FoldGemmaT5 checkpointing
        trainer_t5 = FoldGemmaTrainer(model_type=ModelType.FOLDGEMMA_T5)
        trainer_t5.initialize(456)
        trainer_t5.save_checkpoint(tmp_dir + "_t5")
        trainer_t5.load_checkpoint(tmp_dir + "_t5")
