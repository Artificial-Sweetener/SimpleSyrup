# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load an installed Anima model and selected regional LoRA fixture."""

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
    """Retain one loaded model and its admitted adapter surface."""

    source: Any
    surface: AnimaModuleSurface
    admission: AnimaLoraAdmission


def load_model_fixture(
    model_artifact: PerformanceArtifact,
    lora_artifact: PerformanceArtifact,
) -> AnimaPerformanceModelFixture:
    """Load selected artifacts and require a nonempty admitted adapter surface."""

    source: Any = DiffusionModelLoader().load_path(model_artifact.path, "default")
    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(source.model.diffusion_model)
    admission = ANIMA_LORA_TARGET_CLASSIFIER.admit(
        load_file(str(lora_artifact.path), device="cpu")
    )
    if not admission.targets:
        raise ValueError("Anima performance adapter must expose admitted targets.")
    return AnimaPerformanceModelFixture(source, surface, admission)
