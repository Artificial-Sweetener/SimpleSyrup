"""Prepare one verified resident Anima performance suite."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import comfy.model_management
import torch

from .artifacts import require_artifact, verify_artifact
from .execution_fixture import build_attention_fixture, build_input_fixture
from .manifest import PerformanceManifest
from .measurement import activate_measurement_model
from .model_fixture import load_model_fixture
from .profile import build_runtime_profile
from .runtime_profile import PerformanceRuntimeProfile


@dataclass(frozen=True, slots=True)
class PreparedPerformanceSuite:
    """Retain exact ordered profiles and their shared deterministic tensors."""

    manifest: PerformanceManifest
    device: torch.device
    profiles: tuple[PerformanceRuntimeProfile, ...]
    latent: torch.Tensor
    context: torch.Tensor
    sample_sigmas: torch.Tensor


@contextmanager
def performance_suite_session(
    manifest: PerformanceManifest,
) -> Iterator[PreparedPerformanceSuite]:
    """Prepare one suite and always unload its installed model state."""

    try:
        yield prepare_performance_suite(manifest)
    finally:
        comfy.model_management.unload_all_models()


def prepare_performance_suite(
    manifest: PerformanceManifest,
) -> PreparedPerformanceSuite:
    """Verify artifacts, load the source, and derive every declared profile."""

    for artifact in manifest.artifacts:
        verify_artifact(artifact)
    model_artifact = require_artifact(manifest, "anima_base")
    lora_artifact = require_artifact(manifest, "regional_lora")
    fixture = load_model_fixture(model_artifact, lora_artifact)
    device = torch.device(fixture.source.load_device)
    activate_measurement_model(fixture.source)
    attention = build_attention_fixture(manifest, device=device)
    profiles = tuple(
        build_runtime_profile(
            definition,
            source=fixture.source,
            surface=fixture.surface,
            attention=attention,
            admission=fixture.admission,
            lora_path=lora_artifact.path,
        )
        for definition in manifest.profiles
    )
    latent, context, sample_sigmas = build_input_fixture(
        manifest,
        attention,
        device=device,
    )
    return PreparedPerformanceSuite(
        manifest,
        device,
        profiles,
        latent,
        context,
        sample_sigmas,
    )
