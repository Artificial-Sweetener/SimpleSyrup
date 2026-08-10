# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own SimpleSyrup's guarded Comfy safetensors dtype compatibility."""

from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ..domain.model_quantization import QuantizationFormat
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
_FORMAT_DTYPES = {
    QuantizationFormat.FP8_E4M3: frozenset({"F32", "F8_E4M3"}),
    QuantizationFormat.FP8_E5M2: frozenset({"F32", "F8_E5M2"}),
    QuantizationFormat.NVFP4: frozenset({"F32", "F8_E4M3", "U8"}),
    QuantizationFormat.MXFP8: frozenset({"F8_E4M3", "F8_E8M0"}),
}


class ComfySafetensorsDtypeRegistry:
    """Register and validate dtypes used by Comfy's AIMDO mmap loader."""

    def __init__(
        self,
        comfy_utils_module: ModuleType | None = None,
        torch_module: ModuleType | None = None,
    ) -> None:
        """Create a registry with injectable host boundaries."""

        self._comfy_utils_module = comfy_utils_module
        self._torch_module = torch_module or torch

    def register_extension_dtypes(self) -> bool:
        """Register E8M0 when safe, returning whether MXFP8 is loadable."""

        dtype = getattr(self._torch_module, "float8_e8m0fnu", None)
        if dtype is None:
            return False
        types = self._types()
        existing = types.get("F8_E8M0")
        if existing is None:
            types["F8_E8M0"] = dtype
            return True
        if existing == dtype:
            return True
        LOGGER.warning(
            "Comfy safetensors E8M0 dtype mapping conflicts with PyTorch",
            extra={"registered_dtype": str(existing), "expected_dtype": str(dtype)},
        )
        return False

    def supports_format(self, quantization_format: QuantizationFormat) -> bool:
        """Return whether the active Comfy loader maps every format dtype."""

        if quantization_format is QuantizationFormat.MXFP8:
            if not self.register_extension_dtypes():
                return False
        return _FORMAT_DTYPES[quantization_format].issubset(self._types())

    def validate_checkpoint_header(self, checkpoint_path: Path) -> None:
        """Reject generated checkpoints containing host-unreadable dtypes."""

        header = _read_safetensors_header(checkpoint_path)
        used_dtypes = {
            value.get("dtype")
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict)
        }
        unknown = sorted(
            dtype
            for dtype in used_dtypes
            if isinstance(dtype, str) and dtype not in self._types()
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                "Generated checkpoint uses safetensors dtypes unsupported by the "
                f"active Comfy loader: {joined}."
            )

    def _types(self) -> dict[str, object]:
        """Return Comfy's loader dtype map through one isolated private boundary."""

        comfy_utils = self._comfy_utils_module or importlib.import_module("comfy.utils")
        types: Any = getattr(comfy_utils, "_TYPES", None)
        if not isinstance(types, dict):
            raise RuntimeError("Comfy's safetensors dtype registry is unavailable.")
        return types


def register_comfy_safetensors_dtypes() -> None:
    """Register SimpleSyrup's lightweight host dtype compatibility at import."""

    try:
        ComfySafetensorsDtypeRegistry().register_extension_dtypes()
    except (ImportError, RuntimeError) as error:
        LOGGER.warning(
            "Comfy safetensors dtype compatibility registration failed",
            extra={"reason": str(error)},
        )


def _read_safetensors_header(path: Path) -> dict[str, object]:
    """Read only a safetensors JSON header without materializing tensor data."""

    with path.open("rb") as checkpoint:
        header_length_bytes = checkpoint.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(
                "Generated checkpoint has an incomplete safetensors header."
            )
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header_bytes = checkpoint.read(header_length)
    try:
        payload: object = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Generated checkpoint has an invalid safetensors header."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Generated checkpoint safetensors header must be an object.")
    return payload
