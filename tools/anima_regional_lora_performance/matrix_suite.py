# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare and clean one resident P10.1 scaling suite."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import comfy.model_management
import torch

from .artifacts import require_artifact, verify_artifact
from .matrix_fixture import (
    build_matrix_attention_fixture,
    build_matrix_input_fixture,
)
from .matrix_manifest import ScalingPerformanceManifest
from .matrix_profile import PreparedScalingProfile, build_scaling_runtime_profile
from .measurement import activate_measurement_model
from .model_fixture import load_model_fixture


@dataclass(frozen=True, slots=True)
class PreparedScalingSuite:
    """Retain ordered runtimes and their aligned deterministic model inputs."""

    manifest: ScalingPerformanceManifest
    device: torch.device
    profiles: tuple[PreparedScalingProfile, ...]
    contexts: tuple[torch.Tensor, ...]
    latent: torch.Tensor
    sample_sigmas: torch.Tensor

    def __post_init__(self) -> None:
        """Require exact profile/context alignment before measurement."""

        if len(self.profiles) != len(self.contexts):
            raise ValueError("Scaling suite profiles and contexts must align.")
        if tuple(profile.definition for profile in self.profiles) != (
            self.manifest.profiles
        ):
            raise ValueError("Scaling suite profiles must retain manifest order.")


@contextmanager
def scaling_suite_session(
    manifest: ScalingPerformanceManifest,
) -> Iterator[PreparedScalingSuite]:
    """Prepare one complete suite and always unload installed model state."""

    try:
        yield prepare_scaling_suite(manifest)
    finally:
        comfy.model_management.unload_all_models()


def prepare_scaling_suite(
    manifest: ScalingPerformanceManifest,
) -> PreparedScalingSuite:
    """Verify artifacts, load one source, and derive every scaling position."""

    for artifact in manifest.artifacts:
        verify_artifact(artifact)
    model_artifact = require_artifact(manifest, "anima_base")
    lora_artifact = require_artifact(manifest, "regional_lora")
    fixture = load_model_fixture(model_artifact, lora_artifact)
    device = torch.device(fixture.source.load_device)
    activate_measurement_model(fixture.source)
    prepared: list[PreparedScalingProfile] = []
    contexts: list[torch.Tensor] = []
    first_attention = None
    for definition in manifest.profiles:
        attention = build_matrix_attention_fixture(
            manifest,
            definition,
            device=device,
        )
        if first_attention is None:
            first_attention = attention
        prepared.append(
            build_scaling_runtime_profile(
                definition,
                source=fixture.source,
                surface=fixture.surface,
                attention=attention,
                admission=fixture.admission,
                lora_path=lora_artifact.path,
            )
        )
        contexts.append(attention.contexts.base_context)
    if first_attention is None:
        raise AssertionError("Scaling suite requires at least one profile.")
    latent, _, sample_sigmas = build_matrix_input_fixture(
        manifest,
        first_attention,
        device=device,
    )
    return PreparedScalingSuite(
        manifest,
        device,
        tuple(prepared),
        tuple(contexts),
        latent,
        sample_sigmas,
    )
