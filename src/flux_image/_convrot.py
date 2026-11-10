# Copyright 2026 YUZUKI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the official Comfy-format FLUX 3 ConvRot INT8 DiT into Diffusers.

Black Forest Labs publishes the 8 GB INT8 checkpoint as a single ComfyUI
safetensors file.  Diffusers supports FLUX 3, but it does not currently have a
loader for Comfy's ``int8_tensorwise`` + ConvRot layout.  This module bridges
that format without converting or dequantizing the weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn


class ConvRotInt8Linear(nn.Module):
    """A weight-only INT8 linear backed by comfy-kitchen's ConvRot kernel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        convrot_groupsize: int = 256,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.convrot_groupsize = convrot_groupsize
        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("weight_scale", weight_scale.float(), persistent=True)
        self.register_buffer("bias", bias, persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        return ck.int8_linear(
            inputs,
            self.weight,
            self.weight_scale,
            self.bias,
            self.compute_dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, convrot_groupsize={self.convrot_groupsize}"
        )


class ConvRotLoRALinear(nn.Module):
    """INT8 ConvRot linear with a BF16 low-rank residual adapter."""

    def __init__(self, base: ConvRotInt8Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, strength: float):
        super().__init__()
        self.base = base
        self.strength = float(strength)
        self.register_buffer("lora_a", lora_a, persistent=False)
        self.register_buffer("lora_b", lora_b, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        residual = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs.to(self.lora_a.dtype), self.lora_a), self.lora_b
        )
        return result + residual.to(result.dtype) * self.strength

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def _quant_target_names(source_name: str) -> list[str]:
    """Map an original FLUX 3 linear name to one or three Diffusers names."""
    name = source_name
    if name.startswith("content_mode_blocks."):
        name = "mode_blocks." + name.removeprefix("content_mode_blocks.")
    elif name.startswith("single_blocks."):
        name = "single_transformer_blocks." + name.removeprefix("single_blocks.")

    suffixes = {
        ".attn_out": ".attn.to_out.0",
        ".mlp_in": ".ff.linear_in",
        ".mlp_out": ".ff.linear_out",
    }
    for old, new in suffixes.items():
        if name.endswith(old):
            return [name[: -len(old)] + new]
    if name.endswith(".qkv"):
        prefix = name[: -len(".qkv")] + ".attn."
        return [prefix + "to_q", prefix + "to_k", prefix + "to_v"]
    raise ValueError(f"Unsupported quantized FLUX 3 tensor: {source_name}")


def _replace_module(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, name, module)


def load_flux3_convrot_transformer(
    checkpoint: str | Path,
    config_dir: str | Path,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load BFL's official distilled ConvRot INT8 transformer on CPU.

    The returned module keeps INT8 weights and FP32 row scales intact. Move it
    to CUDA without a dtype argument so the FP32 scales are not downcast.
    """
    from diffusers import Flux3Transformer2DModel
    from diffusers.loaders.single_file_utils import (
        convert_flux3_transformer_checkpoint_to_diffusers,
    )

    checkpoint = Path(checkpoint)
    config_dir = Path(config_dir)
    config = Flux3Transformer2DModel.load_config(config_dir)
    with init_empty_weights():
        model = Flux3Transformer2DModel.from_config(config)

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())

        quant_bases = sorted(
            key.removesuffix(".comfy_quant") for key in keys if key.endswith(".comfy_quant")
        )
        for source_base in quant_bases:
            quant_config = json.loads(bytes(tensors.get_tensor(source_base + ".comfy_quant").tolist()))
            if quant_config.get("format") != "int8_tensorwise" or not quant_config.get("convrot"):
                raise ValueError(f"Unsupported quantization config for {source_base}: {quant_config}")

            weight = tensors.get_tensor(source_base + ".weight")
            weight_scale = tensors.get_tensor(source_base + ".weight_scale")
            targets = _quant_target_names(source_base)
            weights = weight.chunk(len(targets), dim=0)
            scales = weight_scale.chunk(len(targets), dim=0)
            for target, target_weight, target_scale in zip(targets, weights, scales, strict=True):
                old = model.get_submodule(target)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at {target}, found {type(old).__name__}")
                if tuple(target_weight.shape) != (old.out_features, old.in_features):
                    raise ValueError(
                        f"Shape mismatch for {target}: checkpoint {tuple(target_weight.shape)}, "
                        f"model {(old.out_features, old.in_features)}"
                    )
                _replace_module(
                    model,
                    target,
                    ConvRotInt8Linear(
                        old.in_features,
                        old.out_features,
                        target_weight.contiguous(),
                        target_scale.contiguous(),
                        compute_dtype=compute_dtype,
                        convrot_groupsize=int(quant_config.get("convrot_groupsize", 256)),
                    ),
                )

        floating = {
            key: tensors.get_tensor(key)
            for key in keys
            if tensors.get_slice(key).get_dtype() in {"BF16", "F16", "F32"}
            and not key.endswith(".weight_scale")
        }

    floating = convert_flux3_transformer_checkpoint_to_diffusers(floating)
    expected = dict(model.named_parameters())
    missing = set(expected) - set(floating)
    unexpected = set(floating) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"FLUX 3 floating tensor mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, value in floating.items():
        set_module_tensor_to_device(model, name, "cpu", value=value, dtype=compute_dtype)

    meta = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"FLUX 3 loader left meta tensors: {meta[:10]}")
    model.eval()
    return model
