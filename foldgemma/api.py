"""FoldGemma unified API."""
from __future__ import annotations

from typing import Any

from foldgemma.config import FoldGemmaConfig, ModelType

try:
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx

    from foldgemma.data.pipeline import FoldGemmaDataPipeline
    from foldgemma.export import FoldGemmaExporter
    from foldgemma.train.models.foldgemma import FoldGemma as FlaxFoldGemma
    from foldgemma.train.models.foldgemma_t5 import FoldGemmaT5 as FlaxFoldGemmaT5

    TRAIN_AVAILABLE = True
except ImportError as e:
    TRAIN_AVAILABLE = False
    TRAIN_ERROR = e

try:
    import torch

    from foldgemma.inference.models.foldgemma import FoldGemma as TorchFoldGemma
    from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as TorchFoldGemmaT5

    INFERENCE_AVAILABLE = True
except ImportError as e:
    INFERENCE_AVAILABLE = False
    INFERENCE_ERROR = e


class FoldGemmaTrainer:
    """JAX/Flax NNX API for training FoldGemma."""

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
        self.checkpointer = ocp.StandardCheckpointer()
        self.model: nnx.Module | None = None
        self.optimizer: nnx.Optimizer | None = None
        self.step: int = 0

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create the Adafactor optimizer."""
        return optax.adafactor(learning_rate=self.learning_rate)

    def initialize(self, seed: int | None = None) -> None:
        """Initialize the model weights and Optimizer."""
        if self.model is not None and self.optimizer is not None:
            return

        current_seed = seed if seed is not None else self.seed
        rngs = nnx.Rngs(current_seed)
        
        if self.config.model_type == ModelType.FOLDGEMMA_T5:
            self.model = FlaxFoldGemmaT5(self.config, rngs=rngs)
        else:
            self.model = FlaxFoldGemma(self.config, rngs=rngs)

        tx = self._create_optimizer()
        self.optimizer = nnx.Optimizer(self.model, tx)
        self.step = 0

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load state from an orbax checkpoint."""
        import typing

        # Initialize with dummy parameters if not already initialized
        if self.model is None or self.optimizer is None:
            self.initialize()

        # Orbax checkpoint restoration
        ckpt_manager = ocp.CheckpointManager(checkpoint_dir)
        step = ckpt_manager.latest_step()
        if step is not None:
            # Create a combined state tuple for saving/restoring
            _, state = nnx.split((self.model, self.optimizer))
            restored = ckpt_manager.restore(step, args=ocp.args.StandardRestore(state))
            nnx.update((self.model, self.optimizer), restored)
            self.step = step

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save state to an orbax checkpoint."""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Cannot save checkpoint before initialization.")

        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        with ocp.CheckpointManager(checkpoint_dir, options=options) as ckpt_manager:
            _, state = nnx.split((self.model, self.optimizer))
            ckpt_manager.save(self.step, args=ocp.args.StandardSave(state))

    def fit(
        self,
        pipeline: FoldGemmaDataPipeline,
        epochs: int = 1,
        steps_per_epoch: int = 10,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Train the model using the provided data pipeline."""
        if self.model is None or self.optimizer is None:
            self.initialize()

        from foldgemma.train.train import train_step

        print(f"Starting training for {epochs} epochs...")
        dataset = pipeline.get_train_dataset()
        iterator = iter(dataset)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Keep epoch loss as a JAX array to prevent host-device blocking
            epoch_loss = jnp.array(0.0)

            for step in range(steps_per_epoch):
                try:
                    batch_tf = next(iterator)
                except StopIteration:
                    iterator = iter(dataset)
                    batch_tf = next(iterator)

                # Convert tf.Tensor to numpy arrays for jax
                batch = {
                    "inputs": jnp.array(batch_tf["inputs"].numpy()),
                    "targets": jnp.array(batch_tf["targets"].numpy()),
                    "plddt": jnp.array(batch_tf["plddt"].numpy()),
                }
                if self.config.model_type == ModelType.FOLDGEMMA_T5:
                    batch["decoder_input_ids"] = batch["targets"]

                loss = train_step(
                    self.model,
                    self.optimizer,
                    batch,
                    pad_id=pipeline.vocabulary.pad_id,
                    unk_id=pipeline.vocabulary.unk_id,
                    plddt_threshold=70.0,
                )
                epoch_loss += loss
                self.step += 1

                if (step + 1) % 10 == 0:
                    # Only block the host to log every N steps
                    current_loss = float(loss)
                    print(f"  Step {step + 1}/{steps_per_epoch} - Loss: {current_loss:.4f}")

            # Block once at the end of the epoch
            avg_loss = float(epoch_loss) / steps_per_epoch
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

    def export_to_pytorch(self, output_path: str) -> torch.nn.Module:
        """Export the current JAX weights to PyTorch safe tensors format."""
        if self.model is None:
            raise RuntimeError("Cannot export uninitialized model.")
        
        # Get raw dict from nnx model
        _, state = nnx.split(self.model, nnx.Param)
        flax_params = state.to_pure_dict()
        
        exporter = FoldGemmaExporter(self.config)
        return exporter.export_to_pytorch(flax_params, output_path)

class FoldGemmaInference:
    """PyTorch API for FoldGemma inference."""

    def __init__(
        self,
        config: FoldGemmaConfig | None = None,
        compile_model: bool = True,
        model_type: ModelType | str | None = None,
    ):
        if not INFERENCE_AVAILABLE:
            raise ImportError(
                "Inference dependencies are missing. Please install them using "
                "`pip install foldgemma[inference]` "
                "or `uv add foldgemma[inference]`."
            ) from INFERENCE_ERROR

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

        if self.config.model_type == ModelType.FOLDGEMMA_T5:
            self.model: torch.nn.Module = TorchFoldGemmaT5(self.config)
        else:
            self.model = TorchFoldGemma(self.config)

        # Determine best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Cast to bfloat16 and move to device for maximum throughput
        self.model.to(dtype=torch.bfloat16, device=self.device)

        if compile_model and hasattr(torch, "compile"):
            print("Compiling PyTorch model with torch.compile...")
            import typing

            compiled_model = torch.compile(self.model, mode="max-autotune")
            self.model = typing.cast(torch.nn.Module, compiled_model)

    def load_weights(self, safetensors_path: str) -> None:
        """Load PyTorch weights from a safetensors file."""
        import torch
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)
        # Convert state_dict to bfloat16 to match the model
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(state_dict)
        raw_model.eval()

    def predict(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run inference forward pass."""
        import torch

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ) if self.device.type != "mps" else torch.autocast(device_type="cpu", enabled=False),
        ):
            input_ids = input_ids.to(self.device)
            # Move any other tensor kwargs to the device
            kwargs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return self.model(input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run autoregressive generation (for FoldGemmaT5)."""
        import typing

        import torch

        raw_model = getattr(self.model, "_orig_mod", self.model)
        generate_fn = getattr(raw_model, "generate", None)
        if generate_fn is None or not callable(generate_fn):
            raise AttributeError(
                f"Model type {self.config.model_type} does not support generation."
            )

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ) if self.device.type != "mps" else torch.autocast(device_type="cpu", enabled=False),
        ):
            input_ids = input_ids.to(self.device)
            # Move any other tensor kwargs to the device
            kwargs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return typing.cast(torch.Tensor, generate_fn(input_ids, **kwargs))
