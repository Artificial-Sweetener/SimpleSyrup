"""Coordinate the real full-resolution Anima performance matrix."""

from __future__ import annotations

import torch

from .environment import performance_environment
from .manifest import PerformanceManifest
from .measurement import activate_measurement_model, execute_call_sequence
from .results import PerformanceObservation
from .suite import performance_suite_session


class AnimaRegionalLoraPerformanceRunner:
    """Coordinate verified fixture loading, warmup, repeats, and cleanup."""

    def run(
        self,
        manifest: PerformanceManifest,
    ) -> tuple[tuple[PerformanceObservation, ...], dict[str, object]]:
        """Measure every declared profile and publish installed environment data."""

        if not torch.cuda.is_available():
            raise RuntimeError("Anima performance requires an available CUDA device.")
        with performance_suite_session(manifest) as suite:
            for profile in suite.profiles:
                activate_measurement_model(profile.model)
                execute_call_sequence(
                    profile,
                    latent=suite.latent,
                    context=suite.context,
                    sample_sigmas=suite.sample_sigmas,
                    call_count=manifest.warmup_calls,
                    measured=False,
                )
            observations: list[PerformanceObservation] = []
            for repeat_index in range(manifest.repeats):
                for profile in suite.profiles:
                    activate_measurement_model(profile.model)
                    measured = execute_call_sequence(
                        profile,
                        latent=suite.latent,
                        context=suite.context,
                        sample_sigmas=suite.sample_sigmas,
                        call_count=manifest.denoiser_calls,
                        measured=True,
                    )
                    observations.append(
                        PerformanceObservation(
                            profile_id=profile.definition.profile_id,
                            repeat_index=repeat_index,
                            runtime_ms=measured.runtime_ms,
                            peak_vram_bytes=measured.peak_vram_bytes,
                            denoiser_calls=measured.denoiser_calls,
                            output_sha256=measured.output_sha256,
                            prepared_target_count=profile.prepared_target_count,
                        )
                    )
            return tuple(observations), performance_environment(suite.device)


ANIMA_REGIONAL_LORA_PERFORMANCE_RUNNER = AnimaRegionalLoraPerformanceRunner()
