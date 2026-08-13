# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate ordered isolated P10.1 VRAM profile measurement."""

from __future__ import annotations

import torch

from .environment import performance_environment
from .isolated_measurement import (
    IsolatedProfileMeasurement,
    IsolatedVramObservation,
    complete_isolated_observation,
    current_cuda_allocation,
    measure_isolated_profile,
)
from .isolated_suite import (
    ValidatedScalingArtifacts,
    isolated_profile_session,
    release_isolated_device,
    stabilize_isolated_cuda_runtime,
    validate_scaling_artifacts,
)
from .matrix_manifest import ScalingPerformanceManifest, ScalingPerformanceProfile
from .measurement import activate_measurement_model, execute_call_sequence


class AnimaRegionalLoraIsolatedVramRunner:
    """Visit every declared profile in a separate released model lifetime."""

    def run(
        self,
        manifest: ScalingPerformanceManifest,
    ) -> tuple[tuple[IsolatedVramObservation, ...], dict[str, object]]:
        """Measure all profiles after one artifact verification pass."""

        if not torch.cuda.is_available():
            raise RuntimeError("Isolated Anima VRAM measurement requires CUDA.")
        artifacts = validate_scaling_artifacts(manifest)
        device = torch.device("cuda", torch.cuda.current_device())
        stabilize_isolated_cuda_runtime(device)
        observations: list[IsolatedVramObservation] = []
        for definition in manifest.profiles:
            release_isolated_device(device)
            allocation_before = current_cuda_allocation(device)
            try:
                measurement = self._measure_profile(
                    manifest,
                    definition,
                    artifacts,
                    expected_device=device,
                )
            finally:
                release_isolated_device(device)
            allocation_after = current_cuda_allocation(device)
            observations.append(
                complete_isolated_observation(
                    definition,
                    measurement,
                    allocation_before_load_bytes=allocation_before,
                    allocation_after_cleanup_bytes=allocation_after,
                )
            )
        return tuple(observations), performance_environment(device)

    def _measure_profile(
        self,
        manifest: ScalingPerformanceManifest,
        definition: ScalingPerformanceProfile,
        artifacts: ValidatedScalingArtifacts,
        *,
        expected_device: torch.device,
    ) -> IsolatedProfileMeasurement:
        """Return CPU-only evidence after one profile session has exited."""

        with isolated_profile_session(manifest, definition, artifacts) as prepared:
            if prepared.device != expected_device:
                raise ValueError(
                    "Isolated profile loaded on an unexpected CUDA device."
                )
            activate_measurement_model(prepared.profile.runtime.model)
            execute_call_sequence(
                prepared.profile.runtime,
                latent=prepared.latent,
                context=prepared.context,
                sample_sigmas=prepared.sample_sigmas,
                call_count=manifest.warmup_calls,
                measured=False,
            )
            return measure_isolated_profile(
                prepared.profile,
                latent=prepared.latent,
                context=prepared.context,
                sample_sigmas=prepared.sample_sigmas,
                call_count=manifest.denoiser_calls,
            )


ANIMA_REGIONAL_LORA_ISOLATED_VRAM_RUNNER = AnimaRegionalLoraIsolatedVramRunner()
