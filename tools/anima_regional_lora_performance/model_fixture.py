"""Load the exact installed Anima model and regional LoRA fixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safetensors.torch import load_file

from simple_syrup.runtime.diffusion_model_loader import DiffusionModelLoader
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_LORA_TARGET_CLASSIFIER,
    AnimaLoraAdmission,
)

from .manifest import PerformanceArtifact


@dataclass(frozen=True, slots=True)
class AnimaPerformanceModelFixture:
    """Retain one loaded model and exact admitted ADAPTER_A surface."""

    source: Any
    surface: AnimaModuleSurface
    admission: AnimaLoraAdmission


def load_model_fixture(
    model_artifact: PerformanceArtifact,
    lora_artifact: PerformanceArtifact,
) -> AnimaPerformanceModelFixture:
    """Load installed artifacts and require full pinned adapter fidelity."""

    source: Any = DiffusionModelLoader().load_path(model_artifact.path, "default")
    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(source.model.diffusion_model)
    admission = ANIMA_LORA_TARGET_CLASSIFIER.admit(
        load_file(str(lora_artifact.path), device="cpu")
    )
    _validate_admission(admission)
    return AnimaPerformanceModelFixture(source, surface, admission)


def _validate_admission(admission: AnimaLoraAdmission) -> None:
    """Require every pinned rank-32 ADAPTER_A target before measurement."""

    if len(admission.targets) != 448:
        raise ValueError(
            "Anima performance ADAPTER_A fixture must expose exactly 448 targets."
        )
    if any(target.adapter.rank != 32 for target in admission.targets):
        raise ValueError("Anima performance ADAPTER_A fixture must retain full rank 32.")
