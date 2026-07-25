"""Exporter API for converting JAX/Flax weights to PyTorch."""

import os
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.inference.models.foldgemma import FoldGemma as TorchFoldGemma
from foldgemma.inference.models.foldgemma_t5 import FoldGemmaT5 as TorchFoldGemmaT5


class FoldGemmaExporter:
    """API for exporting JAX/Flax models to PyTorch."""

    def __init__(self, config: FoldGemmaConfig):
        self.config = config

    def convert_weights(self, flax_params: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Convert a Flax parameters dictionary to a PyTorch state_dict."""
        torch_state = {}

        def to_torch(x: Any) -> torch.Tensor:
            import numpy as np

            return torch.from_numpy(np.array(x).copy())

        # Encoder Embeddings
        if "embed_tokens" in flax_params:
            torch_state["embed_tokens.weight"] = to_torch(flax_params["embed_tokens"]["embedding"])

        # Encoder Layers
        if "layers" in flax_params:
            for i in range(self.config.num_hidden_layers):
                if i in flax_params["layers"]:
                    f_layer = flax_params["layers"][i]
                    t_prefix = f"layers.{i}"

                    # Input LayerNorm
                    if "input_layernorm" in f_layer:
                        torch_state[f"{t_prefix}.input_layernorm.scale"] = to_torch(
                            f_layer["input_layernorm"]["scale"]
                        )

                    # Post-Attention LayerNorm
                    if "post_attention_layernorm" in f_layer:
                        torch_state[f"{t_prefix}.post_attention_layernorm.scale"] = to_torch(
                            f_layer["post_attention_layernorm"]["scale"]
                        )

                    # Attention Q, K, V, O projections (transpose needed)
                    if "self_attn" in f_layer:
                        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                            if proj in f_layer["self_attn"]:
                                torch_state[f"{t_prefix}.self_attn.{proj}.weight"] = to_torch(
                                    f_layer["self_attn"][proj]["kernel"]
                                ).T.contiguous()

                    # MLP Gate, Up, Down projections (transpose needed)
                    if "mlp" in f_layer:
                        for proj in ["gate_proj", "up_proj", "down_proj"]:
                            if proj in f_layer["mlp"]:
                                torch_state[f"{t_prefix}.mlp.{proj}.weight"] = to_torch(
                                    f_layer["mlp"][proj]["kernel"]
                                ).T.contiguous()

        # Encoder Final Norm
        if "norm" in flax_params:
            torch_state["norm.scale"] = to_torch(flax_params["norm"]["scale"])

        # Check if model is FoldGemmaT5 or contains decoder parameters
        is_t5 = (
            self.config.model_type == ModelType.FOLDGEMMA_T5
            or "decoder_norm" in flax_params
            or "decoder_embed_tokens" in flax_params
        )

        if is_t5:
            # Decoder Embeddings
            if "decoder_embed_tokens" in flax_params:
                torch_state["decoder_embed_tokens.weight"] = to_torch(
                    flax_params["decoder_embed_tokens"]["embedding"]
                )

            # Decoder Layers
            if "decoder_layers" in flax_params:
                for i in range(self.config.num_hidden_layers):
                    if i in flax_params["decoder_layers"]:
                        f_dec_layer = flax_params["decoder_layers"][i]
                        t_prefix = f"decoder_layers.{i}"

                        # Self-Attention LayerNorm
                        if "input_layernorm" in f_dec_layer:
                            torch_state[f"{t_prefix}.input_layernorm.scale"] = to_torch(
                                f_dec_layer["input_layernorm"]["scale"]
                            )
                        elif "self_attn_norm" in f_dec_layer:
                            torch_state[f"{t_prefix}.input_layernorm.scale"] = to_torch(
                                f_dec_layer["self_attn_norm"]["scale"]
                            )

                        # Self-Attention projections
                        if "self_attn" in f_dec_layer:
                            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                                if proj in f_dec_layer["self_attn"]:
                                    torch_state[f"{t_prefix}.self_attn.{proj}.weight"] = to_torch(
                                        f_dec_layer["self_attn"][proj]["kernel"]
                                    ).T.contiguous()

                        # Cross-Attention LayerNorm
                        if "post_attention_layernorm" in f_dec_layer:
                            torch_state[f"{t_prefix}.cross_attn_layernorm.scale"] = to_torch(
                                f_dec_layer["post_attention_layernorm"]["scale"]
                            )
                        elif "cross_attn_norm" in f_dec_layer:
                            torch_state[f"{t_prefix}.cross_attn_layernorm.scale"] = to_torch(
                                f_dec_layer["cross_attn_norm"]["scale"]
                            )

                        # Cross-Attention projections
                        if "cross_attn" in f_dec_layer:
                            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                                if proj in f_dec_layer["cross_attn"]:
                                    torch_state[f"{t_prefix}.cross_attn.{proj}.weight"] = to_torch(
                                        f_dec_layer["cross_attn"][proj]["kernel"]
                                    ).T.contiguous()

                        # MLP LayerNorm
                        if "pre_feedforward_layernorm" in f_dec_layer:
                            torch_state[f"{t_prefix}.post_attention_layernorm.scale"] = to_torch(
                                f_dec_layer["pre_feedforward_layernorm"]["scale"]
                            )
                        elif "mlp_norm" in f_dec_layer:
                            torch_state[f"{t_prefix}.post_attention_layernorm.scale"] = to_torch(
                                f_dec_layer["mlp_norm"]["scale"]
                            )

                        # MLP projections
                        if "mlp" in f_dec_layer:
                            for proj in ["gate_proj", "up_proj", "down_proj"]:
                                if proj in f_dec_layer["mlp"]:
                                    torch_state[f"{t_prefix}.mlp.{proj}.weight"] = to_torch(
                                        f_dec_layer["mlp"][proj]["kernel"]
                                    ).T.contiguous()

            # Decoder Final Norm
            if "decoder_norm" in flax_params:
                torch_state["decoder_norm.scale"] = to_torch(flax_params["decoder_norm"]["scale"])

        # LM Head (transpose needed)
        if "lm_head" in flax_params:
            kernel = to_torch(flax_params["lm_head"]["kernel"])
            torch_state["lm_head.weight"] = kernel.T.contiguous()

        return torch_state

    def export_to_pytorch(
        self, flax_params: Mapping[str, Any], output_path: str
    ) -> torch.nn.Module:
        """Convert and save weights to safetensors, returning the PyTorch model."""
        torch_state = self.convert_weights(flax_params)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_file(torch_state, output_path)

        if (
            self.config.model_type == ModelType.FOLDGEMMA_T5
            or "decoder_norm.scale" in torch_state
        ):
            torch_model: torch.nn.Module = TorchFoldGemmaT5(self.config)
        else:
            torch_model = TorchFoldGemma(self.config)

        torch_model.load_state_dict(torch_state)
        return torch_model


# Legacy backward compatibility alias
FastProtT5Exporter = FoldGemmaExporter
