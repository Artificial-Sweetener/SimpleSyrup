# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration proof for Simple Load Anima on-demand quantization and reuse."""

from __future__ import annotations

import gc
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from simple_syrup.domain.model_quantization import (
    ModelQuantizationRecipe,
    QuantizationProfile,
)
from simple_syrup.domain.quant_cache import SourceCheckpointIdentity
from simple_syrup.nodes.simple_load_anima import SimpleLoadAnima
from simple_syrup.runtime.checkpoint_quantizer import (
    CheckpointQuantizationResult,
    SafetensorsCheckpointQuantizer,
)
from simple_syrup.runtime.diffusion_model_loader import DiffusionModelLoader
from simple_syrup.runtime.quant_cache_leases import QuantCacheLeaseRegistry
from simple_syrup.runtime.quant_cache_repository import QuantCacheRepository
from simple_syrup.runtime.quantization_capabilities import (
    QuantizationCapabilityCatalog,
)
from simple_syrup.runtime.quantization_progress import QuantizationProgressReporter
from simple_syrup.services.anima_diffusion_model_service import (
    AnimaDiffusionModelService,
)
from simple_syrup.services.anima_loader_service import AnimaLoaderService
from simple_syrup.services.quantized_model_resolver import QuantizedModelResolver


class WorkflowFolderPaths(ModuleType):
    """Resolve a complete tiny workflow fixture below one models directory."""

    def __init__(self, models_dir: Path) -> None:
        """Create folder mappings for diffusion, text encoder, VAE, and embeddings."""

        super().__init__("folder_paths")
        self.models_dir = str(models_dir)

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Return one conventional model path."""

        return str(Path(self.models_dir) / folder_name / filename)

    def get_folder_paths(self, folder_name: str) -> list[str]:
        """Return one conventional model directory."""

        return [str(Path(self.models_dir) / folder_name)]


class UnusedAutoResolver:
    """Fail if manual workflow choices unexpectedly trigger auto resolution."""

    def resolve(self, artifact: object, progress: object = None) -> object:
        """Reject unexpected automatic resolution."""

        del artifact, progress
        raise AssertionError("Manual workflow fixtures must not auto-resolve models.")


class FakeUnderlyingModel:
    """Weak-referenceable owner for cache lease verification."""


class FakeModelPatcher:
    """Represent the ComfyUI model returned from a resolved checkpoint path."""

    def __init__(self, loaded_path: str) -> None:
        """Retain the runtime path while exposing an underlying model owner."""

        self.loaded_path = loaded_path
        self.model = FakeUnderlyingModel()


class FakeVAE:
    """Accept the same construction and validation calls as ComfyUI's VAE."""

    def __init__(self, sd: object, metadata: object = None) -> None:
        """Accept decoded VAE state."""

        del sd, metadata

    def throw_exception_if_invalid(self) -> None:
        """Accept the tiny workflow fixture."""


class FixedLimitProvider:
    """Keep the tiny integration cache beneath a generous test limit."""

    def limit_bytes(self) -> int:
        """Return one GiB."""

        return 1024**3


class CountingQuantizer:
    """Count calls while delegating real checkpoint conversion to ComfyUI."""

    def __init__(self) -> None:
        """Create the real quantizer and an empty call counter."""

        self.calls = 0
        self._quantizer = SafetensorsCheckpointQuantizer()

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
        """Delegate one real conversion and increment the call count."""

        self.calls += 1
        return self._quantizer.quantize(
            source=source,
            destination_path=destination_path,
            profile=profile,
            recipe=recipe,
            progress=progress,
            progress_base=progress_base,
            progress_total=progress_total,
        )


def test_simple_load_anima_generates_and_reuses_native_nvfp4_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node execution generates once, reports progress, and then loads its cache."""

    import comfy.model_management as model_management
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils

    device = model_management.get_torch_device()
    if not model_management.supports_nvfp4_compute(device):
        pytest.skip("NVFP4 compute is unavailable on this GPU.")

    models_dir = tmp_path / "models"
    diffusion_path = models_dir / "diffusion_models" / "Anima" / "anima.safetensors"
    diffusion_path.parent.mkdir(parents=True)
    save_file(
        {
            "net.blocks.2.attn.q_proj.weight": torch.randn(
                16, 16, dtype=torch.bfloat16
            ),
            "net.blocks.0.attn.q_proj.weight": torch.randn(
                16, 16, dtype=torch.bfloat16
            ),
        },
        str(diffusion_path),
    )
    folder_paths = WorkflowFolderPaths(models_dir)
    loaded_paths: list[str] = []
    progress_updates: list[tuple[int, int | None]] = []

    def load_diffusion_model(
        path: str, model_options: dict[str, object]
    ) -> FakeModelPatcher:
        """Record the derived checkpoint loaded by the node."""

        assert model_options == {}
        loaded_paths.append(path)
        return FakeModelPatcher(path)

    class FakeProgressBar:
        """Record node progress published during the first conversion."""

        def __init__(self, total: int) -> None:
            """Retain the expected total through later updates."""

            self.total = total

        def update_absolute(self, value: int, total: int | None = None) -> None:
            """Record one absolute progress update."""

            progress_updates.append((value, total))

    monkeypatch.setattr(comfy_sd, "load_diffusion_model", load_diffusion_model)
    monkeypatch.setattr(comfy_sd, "load_clip", lambda **kwargs: "clip")
    monkeypatch.setattr(comfy_sd, "VAE", FakeVAE)
    monkeypatch.setattr(
        comfy_utils,
        "load_torch_file",
        lambda path, return_metadata=False: ({}, {}) if return_metadata else {},
    )
    monkeypatch.setattr(comfy_utils, "ProgressBar", FakeProgressBar)

    cache_repository = QuantCacheRepository(models_dir / "SyrupQuants")
    leases = QuantCacheLeaseRegistry()
    quantizer = CountingQuantizer()
    resolver = QuantizedModelResolver(
        repository=cache_repository,
        quantizer=quantizer,
        capabilities=QuantizationCapabilityCatalog(),
        limit_provider=FixedLimitProvider(),
        leases=leases,
    )
    diffusion_service = AnimaDiffusionModelService(
        DiffusionModelLoader(folder_paths),
        resolver,
        leases,
    )
    service = AnimaLoaderService(
        resolver=UnusedAutoResolver(),  # type: ignore[arg-type]
        diffusion_service=diffusion_service,
        folder_paths_module=folder_paths,
    )
    original_service = SimpleLoadAnima._service
    SimpleLoadAnima._service = service
    try:
        first = SimpleLoadAnima().load_models(
            diffusion_model="Anima/anima.safetensors",
            quantization="nvfp4-mixed",
            diffusion_weight_dtype="fp8_e4m3fn_fast",
            text_encoder="manual_clip.safetensors",
            text_encoder_device="default",
            vae="manual_vae.safetensors",
        )
        first_progress_count = len(progress_updates)
        second = SimpleLoadAnima().load_models(
            diffusion_model="Anima/anima.safetensors",
            quantization="nvfp4-mixed",
            diffusion_weight_dtype="default",
            text_encoder="manual_clip.safetensors",
            text_encoder_device="default",
            vae="manual_vae.safetensors",
        )
    finally:
        SimpleLoadAnima._service = original_service

    assert quantizer.calls == 1
    assert len(loaded_paths) == 2
    assert loaded_paths[0] == loaded_paths[1]
    derived_path = Path(loaded_paths[0])
    assert derived_path.is_relative_to(models_dir / "SyrupQuants")
    assert first_progress_count > 1
    assert len(progress_updates) == first_progress_count
    assert first[1] == "clip"
    assert isinstance(first[2], FakeVAE)
    assert second[1] == "clip"
    assert isinstance(second[2], FakeVAE)
    first_model = cast(Any, first[0])
    assert first_model.simple_syrup_model_provenance["source_model"] == (
        "Anima/anima.safetensors"
    )
    with safe_open(str(derived_path), framework="pt", device="cpu") as checkpoint:
        assert checkpoint.metadata()["simple_syrup.source_model"] == (
            "Anima/anima.safetensors"
        )
        assert "net.blocks.2.attn.q_proj.comfy_quant" in checkpoint.keys()
    assert (models_dir / "SyrupQuants" / "README.txt").is_file()
    del first, second
    gc.collect()
