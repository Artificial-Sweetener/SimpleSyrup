"""Coordinate interleaved warmup and repeats for the P10.1 scaling matrix."""

from __future__ import annotations

import torch

from .environment import performance_environment
from .matrix_manifest import ScalingPerformanceManifest
from .matrix_measurement import (
    ScalingPerformanceObservation,
    measure_scaling_repeat,
)
from .matrix_suite import scaling_suite_session
from .measurement import activate_measurement_model, execute_call_sequence


class AnimaRegionalLoraScalingRunner:
    """Coordinate one resident suite without owning measurement policy."""

    def run(
        self,
        manifest: ScalingPerformanceManifest,
    ) -> tuple[tuple[ScalingPerformanceObservation, ...], dict[str, object]]:
        """Warm every profile, interleave repeats, and return environment data."""

        if not torch.cuda.is_available():
            raise RuntimeError("Anima scaling performance requires a CUDA device.")
        with scaling_suite_session(manifest) as suite:
            for profile, context in zip(
                suite.profiles,
                suite.contexts,
                strict=True,
            ):
                activate_measurement_model(profile.runtime.model)
                execute_call_sequence(
                    profile.runtime,
                    latent=suite.latent,
                    context=context,
                    sample_sigmas=suite.sample_sigmas,
                    call_count=manifest.warmup_calls,
                    measured=False,
                )
            observations: list[ScalingPerformanceObservation] = []
            for repeat_index in range(manifest.repeats):
                for profile, context in zip(
                    suite.profiles,
                    suite.contexts,
                    strict=True,
                ):
                    activate_measurement_model(profile.runtime.model)
                    observations.append(
                        measure_scaling_repeat(
                            profile,
                            repeat_index=repeat_index,
                            latent=suite.latent,
                            context=context,
                            sample_sigmas=suite.sample_sigmas,
                            call_count=manifest.denoiser_calls,
                        )
                    )
            return tuple(observations), performance_environment(suite.device)


ANIMA_REGIONAL_LORA_SCALING_RUNNER = AnimaRegionalLoraScalingRunner()
