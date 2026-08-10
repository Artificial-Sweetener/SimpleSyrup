# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stream source safetensors through ComfyUI-native quantization layouts."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..domain.model_quantization import (
    ModelQuantizationRecipe,
    QuantizationFormat,
    QuantizationProfile,
    TensorDescriptor,
)
from ..domain.quant_cache import SourceCheckpointIdentity
from .comfy_safetensors_dtypes import ComfySafetensorsDtypeRegistry
from .quantization_progress import QuantizationProgressReporter

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class CheckpointQuantizationResult:
    """Summarize a completed checkpoint conversion."""

    quantized_tensor_count: int
    preserved_tensor_count: int
    output_size_bytes: int


class SafetensorsCheckpointQuantizer:
    """Quantize eligible tensors without constructing the source model graph."""

    def __init__(
        self,
        dtype_registry: ComfySafetensorsDtypeRegistry | None = None,
    ) -> None:
        """Create a quantizer with an injectable checkpoint compatibility owner."""

        self._dtype_registry = dtype_registry or ComfySafetensorsDtypeRegistry()

    def quantize(
        self,
        *,
        source: SourceCheckpointIdentity,
        destination_path: Path,
        profile: QuantizationProfile,
        recipe: ModelQuantizationRecipe,
        progress: QuantizationProgressReporter,
        progress_base: int,
        progress_total: int,
    ) -> CheckpointQuantizationResult:
        """Convert one safetensors checkpoint tensor by tensor."""

        if source.path.suffix.lower() not in (".safetensors", ".sft"):
            raise ValueError(
                "On-demand quantization requires a safetensors diffusion model. "
                f"Selected source: '{source.path.name}'."
            )
        if profile.is_original:
            raise ValueError("Original profiles do not require quantization.")

        quant_ops: Any = importlib.import_module("comfy.quant_ops")
        model_management: Any = importlib.import_module("comfy.model_management")
        layouts = _resolve_layouts(quant_ops, profile)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        device = model_management.get_torch_device()
        output: dict[str, torch.Tensor] = {}
        quantized_count = 0
        preserved_count = 0
        processed_bytes = 0

        with safe_open(
            str(source.path), framework="pt", device="cpu"
        ) as source_checkpoint:
            keys = list(source_checkpoint.keys())
            if any(key.endswith(".comfy_quant") for key in keys):
                raise ValueError(
                    "The selected source already contains ComfyUI quantization "
                    "metadata. Choose the original full-precision Anima checkpoint "
                    "as the source."
                )
            metadata = dict(source_checkpoint.metadata() or {})
            with torch.inference_mode():
                for key in keys:
                    tensor_slice = source_checkpoint.get_slice(key)
                    descriptor = TensorDescriptor(
                        name=key,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype_name=tensor_slice.get_dtype(),
                    )
                    tensor_bytes = _storage_bytes(descriptor)
                    tensor = source_checkpoint.get_tensor(key)
                    quantization_format = recipe.policy_for(descriptor, profile)
                    if quantization_format is not None:
                        device_tensor = tensor.to(device=device).contiguous()
                        quantized = quant_ops.QuantizedTensor.from_float(
                            device_tensor,
                            layouts[quantization_format],
                            scale="recalculate",
                        )
                        state_tensors = quantized.state_dict(key)
                        for output_key, output_tensor in state_tensors.items():
                            output[output_key] = (
                                output_tensor.detach().to(device="cpu").contiguous()
                            )
                        layer_name = key.removesuffix(".weight")
                        output[f"{layer_name}.comfy_quant"] = torch.tensor(
                            list(
                                json.dumps(
                                    {"format": quantization_format.value},
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ),
                            dtype=torch.uint8,
                        )
                        quantized_count += 1
                        del quantized, device_tensor
                    else:
                        output[key] = tensor
                        preserved_count += 1
                    processed_bytes += tensor_bytes
                    progress.advance(
                        min(progress_base + processed_bytes, progress_total),
                        progress_total,
                    )

            metadata.update(
                {
                    "simple_syrup.derived_model": "true",
                    "simple_syrup.model_family": recipe.model_family,
                    "simple_syrup.quantization_profile": profile.profile_id,
                    "simple_syrup.profile_label": profile.label,
                    "simple_syrup.profile_version": str(profile.version),
                    "simple_syrup.quantization_formats": ",".join(
                        sorted(item.value for item in profile.required_formats)
                    ),
                    "simple_syrup.recipe_version": str(recipe.version),
                    "simple_syrup.source_model": source.display_name,
                    "simple_syrup.source_sha256": source.sha256,
                }
            )
            save_file(output, str(destination_path), metadata=metadata)

        self._validate_output(destination_path, profile, quantized_count)
        return CheckpointQuantizationResult(
            quantized_tensor_count=quantized_count,
            preserved_tensor_count=preserved_count,
            output_size_bytes=destination_path.stat().st_size,
        )

    def _validate_output(
        self,
        path: Path,
        profile: QuantizationProfile,
        expected_quantized_tensors: int,
    ) -> None:
        """Validate generated dtypes and Comfy markers before cache publication."""

        if expected_quantized_tensors <= 0:
            raise ValueError(
                "The selected checkpoint contained no tensors eligible for the model's "
                "quantization recipe."
            )
        self._dtype_registry.validate_checkpoint_header(path)
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            marker_keys = tuple(
                key for key in checkpoint.keys() if key.endswith(".comfy_quant")
            )
            if len(marker_keys) != expected_quantized_tensors:
                raise ValueError(
                    "Quantized checkpoint validation found incomplete layer metadata."
                )
            expected_formats = ",".join(
                sorted(item.value for item in profile.required_formats)
            )
            metadata = checkpoint.metadata() or {}
            if (
                metadata.get("simple_syrup.quantization_profile") != profile.profile_id
                or metadata.get("simple_syrup.profile_version") != str(profile.version)
                or metadata.get("simple_syrup.quantization_formats") != expected_formats
            ):
                raise ValueError(
                    "Quantized checkpoint validation found inconsistent "
                    "profile metadata."
                )
            for marker_key in marker_keys:
                quantization_format = _marker_format(
                    checkpoint.get_tensor(marker_key), marker_key
                )
                if quantization_format not in profile.required_formats:
                    raise ValueError(
                        "Quantized checkpoint contains a layer format outside the "
                        f"selected profile: {quantization_format.label}."
                    )
                if not self._dtype_registry.supports_format(quantization_format):
                    raise ValueError(
                        "Generated checkpoint quantization metadata requires a "
                        f"format unsupported by the active Comfy loader: "
                        f"{quantization_format.label}."
                    )


def _resolve_layouts(
    quant_ops: Any,
    profile: QuantizationProfile,
) -> dict[QuantizationFormat, str]:
    """Resolve every layout required by one profile before conversion starts."""

    layouts: dict[QuantizationFormat, str] = {}
    for quantization_format in profile.required_formats:
        algorithm = quant_ops.QUANT_ALGOS.get(quantization_format.value)
        if not isinstance(algorithm, dict):
            raise ValueError(
                f"ComfyUI did not register {quantization_format.label} quantization."
            )
        layout_name = algorithm.get("comfy_tensor_layout")
        if not isinstance(layout_name, str):
            raise ValueError(
                f"ComfyUI's {quantization_format.label} layout is invalid."
            )
        layouts[quantization_format] = layout_name
    return layouts


def _storage_bytes(tensor: TensorDescriptor) -> int:
    """Return safetensors storage bytes represented by a descriptor."""

    element_bytes = _DTYPE_BYTES.get(tensor.dtype_name)
    if element_bytes is None:
        raise ValueError(
            f"Unsupported safetensors dtype '{tensor.dtype_name}' for '{tensor.name}'."
        )
    elements = 1
    for dimension in tensor.shape:
        elements *= dimension
    return elements * element_bytes


def _marker_format(marker: torch.Tensor, marker_key: str) -> QuantizationFormat:
    """Parse one Comfy quant marker into a supported domain format."""

    try:
        payload: object = json.loads(bytes(marker.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Quantized checkpoint marker '{marker_key}' is malformed."
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("format"), str):
        raise ValueError(
            f"Quantized checkpoint marker '{marker_key}' has no valid format."
        )
    try:
        return QuantizationFormat(payload["format"])
    except ValueError as error:
        raise ValueError(
            f"Quantized checkpoint marker '{marker_key}' names an unknown format."
        ) from error
