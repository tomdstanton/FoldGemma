"""FoldGemma unified API."""
from __future__ import annotations

import os
import logging
from typing import Any, TYPE_CHECKING
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from foldgemma.data.pipeline import FoldGemmaDataPipeline
from safetensors.torch import load_file, save_file

from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.models.foldgemma import FoldGemma
from foldgemma.models.foldgemma_t5 import FoldGemmaT5
from foldgemma.loss import MaskedCrossEntropyLoss

try:
    import tensorflow as tf
    TRAIN_AVAILABLE = True
except ImportError as e:
    TRAIN_AVAILABLE = False
    TRAIN_ERROR = e

class FoldGemmaTrainer:
    """PyTorch API for training FoldGemma."""

    def __init__(
        self,
        config: FoldGemmaConfig | None = None,
        learning_rate: float = 1e-3,
        model_type: ModelType | str | None = None,
        seed: int = 42,
    ):
        if not TRAIN_AVAILABLE:
            raise ImportError(
                "Training dependencies are missing. Please install them using "
                "`pip install foldgemma[train]` "
                "or `uv add foldgemma[train]`."
            ) from TRAIN_ERROR

        base_config = config or FoldGemmaConfig()
        if model_type is not None:
            resolved_type = ModelType(model_type) if isinstance(model_type, str) else model_type
            if base_config.model_type != resolved_type:
                from dataclasses import replace

                self.config = replace(base_config, model_type=resolved_type)
            else:
                self.config = base_config
        else:
            self.config = base_config

        self.seed = seed
        self.learning_rate = learning_rate
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.step: int = 0
        
        # Determine best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def initialize(self, seed: int | None = None) -> None:
        """Initialize the model weights and Optimizer."""
        if self.model is not None and self.optimizer is not None:
            return

        logger.debug("Inside initialize(). Setting seed...")
        current_seed = seed if seed is not None else self.seed
        torch.manual_seed(current_seed)
        
        logger.debug("Instantiating model...")
        if self.config.model_type == ModelType.FOLDGEMMA_T5:
            self.model = FoldGemmaT5(self.config)
        else:
            self.model = FoldGemma(self.config)

        logger.debug(f"Moving model to device {self.device}...")
        self.model.to(self.device)
        if self.device.type == "cuda":
            logger.debug("Synchronizing CUDA...")
            torch.cuda.synchronize()
        
        logger.debug("Instantiating optimizer...")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.step = 0
        logger.debug("initialize() complete.")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load state from a PyTorch checkpoint."""
        if self.model is None or self.optimizer is None:
            self.initialize()
            
        from pathlib import Path
        ckpt_path = Path(checkpoint_dir)
        model_path = ckpt_path / "model.safetensors"
        opt_path = ckpt_path / "optimizer.pt"
        
        if model_path.exists():
            state_dict = load_file(str(model_path))
            self.model.load_state_dict(state_dict)
            
        if opt_path.exists():
            # weights_only=False may be needed for optimizer state if it uses custom classes, but dicts should be fine
            opt_state = torch.load(str(opt_path), weights_only=False)
            self.optimizer.load_state_dict(opt_state["optimizer"])
            self.step = opt_state.get("step", 0)

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save state to a PyTorch checkpoint."""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Cannot save checkpoint before initialization.")
            
        from pathlib import Path
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model_path = ckpt_path / "model.safetensors"
        opt_path = ckpt_path / "optimizer.pt"
        
        save_file(self.model.state_dict(), str(model_path))
        torch.save({"optimizer": self.optimizer.state_dict(), "step": self.step}, str(opt_path))

    def fit(
        self,
        pipeline: FoldGemmaDataPipeline,
        epochs: int = 1,
        steps_per_epoch: int = 10,
        checkpoint_dir: str | None = None,
        on_epoch_start: Optional[Callable[[int, int], None]] = None,
        on_step: Optional[Callable[[int, float], None]] = None,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """Train the model using the provided data pipeline."""
        if self.model is None or self.optimizer is None:
            self.initialize()

        self.model.train()
        logger.info(f"Starting training for {epochs} epochs...")
        logger.debug("Getting train dataset...")
        dataset = pipeline.get_train_dataset()
        logger.debug("Creating iterator...")
        iterator = iter(dataset)

        for epoch in range(epochs):
            if on_epoch_start:
                on_epoch_start(epoch, epochs)
            epoch_loss = 0.0

            for step in range(steps_per_epoch):
                try:
                    if step == 0:
                        logger.debug("Fetching first batch...")
                    batch_tf = next(iterator)
                    if step == 0:
                        logger.debug("First batch fetched successfully.")
                except StopIteration:
                    iterator = iter(dataset)
                    batch_tf = next(iterator)

                # Convert tf.Tensor to numpy arrays, then to torch on device
                inputs = torch.tensor(batch_tf["inputs"].numpy(), device=self.device)
                targets = torch.tensor(batch_tf["targets"].numpy(), device=self.device)
                plddt = torch.tensor(batch_tf["plddt"].numpy(), device=self.device)
                
                self.optimizer.zero_grad()
                
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ) if self.device.type != "mps" else torch.autocast(device_type="cpu", enabled=False):
                    if self.config.model_type == ModelType.FOLDGEMMA_T5:
                        logits = self.model(input_ids=inputs, decoder_input_ids=targets)
                    else:
                        logits = self.model(input_ids=inputs)
                        
                    loss_fn = MaskedCrossEntropyLoss(
                        pad_id=pipeline.vocabulary.pad_id,
                        unk_id=pipeline.vocabulary.unk_id,
                        plddt_threshold=70.0
                    )
                    
                    loss = loss_fn(
                        logits=logits,
                        targets=targets,
                        plddt=plddt
                    )
                    
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                self.step += 1

                if on_step:
                    on_step(step + 1, current_loss)

            avg_loss = epoch_loss / steps_per_epoch
            if on_epoch_end:
                on_epoch_end(epoch, avg_loss)

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")


