"""Prepare and release one isolated P10.1 scaling profile."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import comfy.model_management
import torch

from .artifacts import require_artifact, verify_artifact
from .manifest import PerformanceArtifact
from .matrix_fixture import (
    build_matrix_attention_fixture,
    build_matrix_input_fixture,
)
from .matrix_manifest import ScalingPerformanceManifest, ScalingPerformanceProfile
from .matrix_profile import PreparedScalingProfile, build_scaling_runtime_profile
from .measurement import activate_measurement_model
from .model_fixture import load_model_fixture


@dataclass(frozen=True, slots=True)
class ValidatedScalingArtifacts:
    """Retain the exact model and LoRA artifacts verified once for a run."""

    model: PerformanceArtifact
    lora: PerformanceArtifact

    def __post_init__(self) -> None:
        """Require the two authoritative artifact roles."""

        if self.model.role != "anima_base" or self.lora.role != "regional_lora":
            raise ValueError("Isolated scaling artifacts have invalid roles.")


@dataclass(frozen=True, slots=True)
class PreparedIsolatedScalingProfile:
    """Retain one profile and only its aligned CUDA execution inputs."""

    manifest: ScalingPerformanceManifest
    definition: ScalingPerformanceProfile
    device: torch.device
    profile: PreparedScalingProfile
    context: torch.Tensor
    latent: torch.Tensor
    sample_sigmas: torch.Tensor

    def __post_init__(self) -> None:
        """Require exact declaration and input-device alignment."""

        if self.profile.definition is not self.definition:
            raise ValueError("Isolated scaling profile must retain its declaration.")
        if not any(
            candidate is self.definition for candidate in self.manifest.profiles
        ):
            raise ValueError(
                "Isolated scaling declaration must belong to its manifest."
            )
        tensors = (self.context, self.latent, self.sample_sigmas)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("Isolated scaling inputs must share the profile device.")


def validate_scaling_artifacts(
    manifest: ScalingPerformanceManifest,
) -> ValidatedScalingArtifacts:
    """Verify every declared byte identity once before isolated model loading."""

    for artifact in manifest.artifacts:
        verify_artifact(artifact)
    return ValidatedScalingArtifacts(
        require_artifact(manifest, "anima_base"),
        require_artifact(manifest, "regional_lora"),
    )


@contextmanager
def isolated_profile_session(
    manifest: ScalingPerformanceManifest,
    definition: ScalingPerformanceProfile,
    artifacts: ValidatedScalingArtifacts,
) -> Iterator[PreparedIsolatedScalingProfile]:
    """Load one profile and always request installed model unload afterward."""

    try:
        yield prepare_isolated_profile(manifest, definition, artifacts)
    finally:
        comfy.model_management.unload_all_models()


def prepare_isolated_profile(
    manifest: ScalingPerformanceManifest,
    definition: ScalingPerformanceProfile,
    artifacts: ValidatedScalingArtifacts,
) -> PreparedIsolatedScalingProfile:
    """Load and derive exactly one declared profile with deterministic inputs."""

    if not any(candidate is definition for candidate in manifest.profiles):
        raise ValueError("Isolated scaling profile must belong to the manifest.")
    if artifacts.model is not require_artifact(manifest, "anima_base"):
        raise ValueError("Isolated scaling model artifact is not authoritative.")
    if artifacts.lora is not require_artifact(manifest, "regional_lora"):
        raise ValueError("Isolated scaling LoRA artifact is not authoritative.")
    fixture = load_model_fixture(artifacts.model, artifacts.lora)
    device = torch.device(fixture.source.load_device)
    activate_measurement_model(fixture.source)
    attention = build_matrix_attention_fixture(manifest, definition, device=device)
    profile = build_scaling_runtime_profile(
        definition,
        source=fixture.source,
        surface=fixture.surface,
        attention=attention,
        admission=fixture.admission,
        lora_path=artifacts.lora.path,
    )
    latent, context, sample_sigmas = build_matrix_input_fixture(
        manifest,
        attention,
        device=device,
    )
    if context is not attention.contexts.base_context:
        raise ValueError("Isolated scaling context must retain attention authority.")
    return PreparedIsolatedScalingProfile(
        manifest,
        definition,
        device,
        profile,
        context,
        latent,
        sample_sigmas,
    )


def release_isolated_device(device: torch.device) -> None:
    """Unload installed models and release unreachable CUDA allocations."""

    comfy.model_management.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def stabilize_isolated_cuda_runtime(device: torch.device) -> None:
    """Establish PyTorch's one-time CUDA math workspace before measurement."""

    left = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    right = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    torch.mm(left, right)
    del left, right
    release_isolated_device(device)
