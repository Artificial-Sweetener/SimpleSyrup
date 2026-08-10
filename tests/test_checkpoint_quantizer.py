# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for streaming ComfyUI checkpoint quantization."""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from simple_syrup.domain.anima_quantization import (
    MXFP8_PROFILE,
    NVFP4_MIXED_PROFILE,
    AnimaQuantizationRecipe,
)
from simple_syrup.domain.model_quantization import QuantizationFormat
from simple_syrup.domain.quant_cache import SourceCheckpointIdentity
from simple_syrup.runtime.checkpoint_quantizer import SafetensorsCheckpointQuantizer
from simple_syrup.runtime.comfy_safetensors_dtypes import (
    ComfySafetensorsDtypeRegistry,
)


@dataclass
class RecordingProgress:
    """Record absolute quantization progress updates."""

    updates: list[tuple[int, int]] = field(default_factory=list)

    def start(self, label: str, total: int) -> None:
        """Record no start because the resolver owns operation setup."""

        del label, total

    def advance(self, current: int, total: int) -> None:
        """Record one absolute update."""

        self.updates.append((current, total))

    def finish(self) -> None:
        """Record no finish because the resolver owns completion."""


class FakeQuantizedTensor:
    """Serialize deterministic fake quant tensors through Comfy's boundary."""

    def __init__(self, source: torch.Tensor) -> None:
        """Retain only the source shape needed by the fake serializer."""

        self._shape = source.shape

    @classmethod
    def from_float(
        cls,
        tensor: torch.Tensor,
        layout_name: str,
        **kwargs: object,
    ) -> FakeQuantizedTensor:
        """Accept the same call shape as ComfyUI's QuantizedTensor."""

        assert layout_name == "FakeLayout"
        assert kwargs == {"scale": "recalculate"}
        return cls(tensor)

    def state_dict(self, prefix: str) -> dict[str, torch.Tensor]:
        """Return representative quantized weight and scale tensors."""

        return {
            prefix: torch.zeros(self._shape, dtype=torch.uint8),
            f"{prefix}_scale": torch.ones((), dtype=torch.float32),
        }


class RecordingDtypeRegistry(ComfySafetensorsDtypeRegistry):
    """Record header and marker validation performed before publication."""

    def __init__(self, *, supports_markers: bool = True) -> None:
        """Create an empty validation record."""

        self._supports_markers = supports_markers
        self.validated_paths: list[Path] = []
        self.validated_formats: list[QuantizationFormat] = []

    def validate_checkpoint_header(self, checkpoint_path: Path) -> None:
        """Record the generated checkpoint whose header was validated."""

        self.validated_paths.append(checkpoint_path)

    def supports_format(self, quantization_format: QuantizationFormat) -> bool:
        """Record and accept one generated Comfy marker format."""

        self.validated_formats.append(quantization_format)
        return self._supports_markers


def test_quantizer_streams_eligible_tensors_and_preserves_anima_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only eligible matrices receive Comfy layer markers and scale tensors."""

    source_path = tmp_path / "anima.safetensors"
    destination = tmp_path / "quantized.safetensors"
    save_file(
        {
            "net.blocks.0.attn.q_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "net.blocks.2.attn.q_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "net.blocks.2.attn.q_proj.bias": torch.randn(4, dtype=torch.bfloat16),
        },
        str(source_path),
        metadata={"source": "test"},
    )
    quant_ops = ModuleType("comfy.quant_ops")
    quant_ops.QUANT_ALGOS = {  # type: ignore[attr-defined]
        "float8_e4m3fn": {"comfy_tensor_layout": "FakeLayout"},
        "nvfp4": {"comfy_tensor_layout": "FakeLayout"},
    }
    quant_ops.QuantizedTensor = FakeQuantizedTensor  # type: ignore[attr-defined]
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.quant_ops", quant_ops)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    progress = RecordingProgress()
    source = _identity(source_path)
    dtype_registry = RecordingDtypeRegistry()

    result = SafetensorsCheckpointQuantizer(dtype_registry).quantize(
        source=source,
        destination_path=destination,
        profile=NVFP4_MIXED_PROFILE,
        recipe=AnimaQuantizationRecipe(),
        progress=progress,
        progress_base=source.size_bytes,
        progress_total=source.size_bytes * 2,
    )

    assert result.quantized_tensor_count == 1
    assert result.preserved_tensor_count == 2
    assert progress.updates
    assert dtype_registry.validated_paths == [destination]
    assert dtype_registry.validated_formats == [QuantizationFormat.NVFP4]
    with safe_open(str(destination), framework="pt", device="cpu") as checkpoint:
        assert set(checkpoint.keys()) == {
            "net.blocks.0.attn.q_proj.weight",
            "net.blocks.2.attn.q_proj.bias",
            "net.blocks.2.attn.q_proj.comfy_quant",
            "net.blocks.2.attn.q_proj.weight",
            "net.blocks.2.attn.q_proj.weight_scale",
        }
        assert (
            checkpoint.get_tensor("net.blocks.0.attn.q_proj.weight").dtype
            is torch.bfloat16
        )
        assert checkpoint.metadata() == {
            "source": "test",
            "simple_syrup.derived_model": "true",
            "simple_syrup.model_family": "Anima",
            "simple_syrup.profile_label": "NVFP4 (Mixed)",
            "simple_syrup.profile_version": "3",
            "simple_syrup.quantization_formats": "float8_e4m3fn,nvfp4",
            "simple_syrup.quantization_profile": "nvfp4-mixed",
            "simple_syrup.recipe_version": "2",
            "simple_syrup.source_model": "Anima/anima.safetensors",
            "simple_syrup.source_sha256": source.sha256,
        }


def test_quantizer_rejects_a_quantized_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived checkpoints cannot silently become lossy quantization sources."""

    source_path = tmp_path / "already-quantized.safetensors"
    save_file(
        {
            "net.blocks.2.linear.weight": torch.ones(4, 4),
            "net.blocks.2.linear.comfy_quant": torch.tensor(
                list(b'{"format":"nvfp4"}'), dtype=torch.uint8
            ),
        },
        str(source_path),
    )
    quant_ops = ModuleType("comfy.quant_ops")
    quant_ops.QUANT_ALGOS = {  # type: ignore[attr-defined]
        "float8_e4m3fn": {"comfy_tensor_layout": "FakeLayout"},
        "nvfp4": {"comfy_tensor_layout": "FakeLayout"},
    }
    quant_ops.QuantizedTensor = FakeQuantizedTensor  # type: ignore[attr-defined]
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.quant_ops", quant_ops)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    with pytest.raises(ValueError, match="already contains ComfyUI quantization"):
        SafetensorsCheckpointQuantizer().quantize(
            source=_identity(source_path),
            destination_path=tmp_path / "output.safetensors",
            profile=NVFP4_MIXED_PROFILE,
            recipe=AnimaQuantizationRecipe(),
            progress=RecordingProgress(),
            progress_base=0,
            progress_total=1,
        )


def test_validation_rejects_inconsistent_profile_metadata(tmp_path: Path) -> None:
    """A generated artifact cannot publish under mismatched profile metadata."""

    destination = tmp_path / "mismatched-profile.safetensors"
    save_file(
        {
            "net.blocks.2.attn.q_proj.weight": torch.zeros(4, 4, dtype=torch.uint8),
            "net.blocks.2.attn.q_proj.comfy_quant": torch.tensor(
                list(b'{"format":"nvfp4"}'), dtype=torch.uint8
            ),
        },
        str(destination),
        metadata={
            "simple_syrup.quantization_profile": "wrong-profile",
            "simple_syrup.profile_version": "3",
            "simple_syrup.quantization_formats": "float8_e4m3fn,nvfp4",
        },
    )

    with pytest.raises(ValueError, match="inconsistent profile metadata"):
        SafetensorsCheckpointQuantizer(RecordingDtypeRegistry())._validate_output(
            destination,
            NVFP4_MIXED_PROFILE,
            1,
        )


def test_validation_rejects_marker_unsupported_by_active_loader(
    tmp_path: Path,
) -> None:
    """Publication fails when the active loader cannot read a marker format."""

    destination = tmp_path / "unsupported-marker.safetensors"
    save_file(
        {
            "net.blocks.2.attn.q_proj.weight": torch.zeros(4, 4, dtype=torch.uint8),
            "net.blocks.2.attn.q_proj.comfy_quant": torch.tensor(
                list(b'{"format":"nvfp4"}'), dtype=torch.uint8
            ),
        },
        str(destination),
        metadata={
            "simple_syrup.quantization_profile": "nvfp4-mixed",
            "simple_syrup.profile_version": "3",
            "simple_syrup.quantization_formats": "float8_e4m3fn,nvfp4",
        },
    )

    with pytest.raises(ValueError, match="unsupported by the active Comfy loader"):
        SafetensorsCheckpointQuantizer(
            RecordingDtypeRegistry(supports_markers=False)
        )._validate_output(
            destination,
            NVFP4_MIXED_PROFILE,
            1,
        )


def test_native_nvfp4_generation_uses_comfy_scale_layout(tmp_path: Path) -> None:
    """The installed ComfyUI runtime produces its native NVFP4 checkpoint shape."""

    try:
        import comfy.model_management as model_management
    except ImportError:
        pytest.skip("ComfyUI runtime is unavailable.")
    if not model_management.supports_nvfp4_compute(model_management.get_torch_device()):
        pytest.skip("NVFP4 compute is unavailable on this GPU.")

    source_path = tmp_path / "native-anima.safetensors"
    destination = tmp_path / "native-nvfp4.safetensors"
    save_file(
        {"net.blocks.2.attn.q_proj.weight": torch.randn(16, 16, dtype=torch.bfloat16)},
        str(source_path),
    )
    source = _identity(source_path)

    SafetensorsCheckpointQuantizer().quantize(
        source=source,
        destination_path=destination,
        profile=NVFP4_MIXED_PROFILE,
        recipe=AnimaQuantizationRecipe(),
        progress=RecordingProgress(),
        progress_base=source.size_bytes,
        progress_total=source.size_bytes * 2,
    )

    with safe_open(str(destination), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        assert "net.blocks.2.attn.q_proj.comfy_quant" in keys
        assert "net.blocks.2.attn.q_proj.weight_scale" in keys
        assert "net.blocks.2.attn.q_proj.weight_scale_2" in keys
        assert (
            checkpoint.get_tensor("net.blocks.2.attn.q_proj.weight").dtype
            is torch.uint8
        )

    import comfy.ops
    import comfy.quant_ops

    state_dict = {
        key.removeprefix("net.blocks.2.attn.q_proj."): value
        for key, value in load_file(str(destination)).items()
    }
    operations = comfy.ops.mixed_precision_ops(
        {"mixed_ops": True}, compute_dtype=torch.bfloat16
    )
    layer = operations.Linear(
        16,
        16,
        bias=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    layer.load_state_dict(state_dict, strict=True)

    assert isinstance(layer.weight, comfy.quant_ops.QuantizedTensor)
    assert layer.quant_format == "nvfp4"


def test_native_mxfp8_generation_reloads_through_aimdo(tmp_path: Path) -> None:
    """E8M0 scale tensors reload through Comfy's active AIMDO mmap path."""

    import comfy.model_management as model_management
    import comfy.utils as comfy_utils

    aimdo_control: Any = importlib.import_module("comfy_aimdo.control")
    aimdo_model_mmap: Any = importlib.import_module("comfy_aimdo.model_mmap")

    device = model_management.get_torch_device()
    if not model_management.supports_mxfp8_compute(device):
        pytest.skip("MXFP8 compute is unavailable on this GPU.")
    source_path = tmp_path / "native-mxfp8-source.safetensors"
    destination = tmp_path / "native-mxfp8.safetensors"
    save_file(
        {"net.blocks.14.attn.q_proj.weight": torch.randn(32, 32, dtype=torch.bfloat16)},
        str(source_path),
    )
    source = _identity(source_path)

    SafetensorsCheckpointQuantizer().quantize(
        source=source,
        destination_path=destination,
        profile=MXFP8_PROFILE,
        recipe=AnimaQuantizationRecipe(),
        progress=RecordingProgress(),
        progress_base=source.size_bytes,
        progress_total=source.size_bytes * 2,
    )
    if not aimdo_control.init():
        pytest.skip("comfy-aimdo is unavailable.")
    importlib.reload(aimdo_model_mmap)
    state_dict, metadata = comfy_utils.load_safetensors(str(destination))

    assert metadata["simple_syrup.quantization_profile"] == "mxfp8"
    assert any(tensor.dtype is torch.float8_e8m0fnu for tensor in state_dict.values())


def test_native_recommended_profile_serializes_both_formats(tmp_path: Path) -> None:
    """Recommended profile emits native NVFP4 and FP8 markers in one checkpoint."""

    import comfy.model_management as model_management

    device = model_management.get_torch_device()
    if not model_management.supports_nvfp4_compute(device):
        pytest.skip("NVFP4 compute is unavailable on this GPU.")
    source_path = tmp_path / "native-mixed-source.safetensors"
    destination = tmp_path / "native-mixed.safetensors"
    save_file(
        {
            "net.blocks.14.attn.q_proj.weight": torch.randn(
                32, 32, dtype=torch.bfloat16
            ),
            "net.blocks.14.attn.v_proj.weight": torch.randn(
                32, 32, dtype=torch.bfloat16
            ),
        },
        str(source_path),
    )
    source = _identity(source_path)

    SafetensorsCheckpointQuantizer().quantize(
        source=source,
        destination_path=destination,
        profile=NVFP4_MIXED_PROFILE,
        recipe=AnimaQuantizationRecipe(),
        progress=RecordingProgress(),
        progress_base=source.size_bytes,
        progress_total=source.size_bytes * 2,
    )

    with safe_open(str(destination), framework="pt", device="cpu") as checkpoint:
        q_marker = bytes(
            checkpoint.get_tensor("net.blocks.14.attn.q_proj.comfy_quant").tolist()
        )
        v_marker = bytes(
            checkpoint.get_tensor("net.blocks.14.attn.v_proj.comfy_quant").tolist()
        )
    assert b'"format":"nvfp4"' in q_marker
    assert b'"format":"float8_e4m3fn"' in v_marker


def _identity(path: Path) -> SourceCheckpointIdentity:
    """Create a complete source identity for a tiny test checkpoint."""

    content = path.read_bytes()
    stat = path.stat()
    return SourceCheckpointIdentity(
        display_name="Anima/anima.safetensors",
        path=path.resolve(),
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=hashlib.sha256(content).hexdigest(),
    )
