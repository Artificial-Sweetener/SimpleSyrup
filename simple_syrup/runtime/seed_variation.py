# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file are adapted from AUTOMATIC1111 stable-diffusion-webui.
# See third_party/manifest.toml and third_party/NOTICE.md.

"""Apply deterministic seed variation at ComfyUI's outer sampling boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from comfy.patcher_extension import WrappersMP

from ..domain.seed_variation import SeedVariationSettings
from .model_patcher_mutations import ModelKeyedWrapperMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE

SEED_VARIATION_WRAPPER_KEY = "simple_syrup.seed_variation"
_SLERP_EPSILON = 1e-6


class SeedVariationNoiseInterpolator:
    """Generate and interpolate one variation tensor without global RNG mutation."""

    def interpolate(
        self,
        base_noise: torch.Tensor,
        settings: SeedVariationSettings,
    ) -> torch.Tensor:
        """Return base-to-variation spherical interpolation for one sampling call."""

        self._validate_noise(base_noise)
        if settings.strength == 0.0 or self._is_disabled_noise(base_noise):
            return base_noise

        variation_noise = self._variation_noise_like(
            base_noise,
            settings.variation_seed,
        )
        if settings.strength == 1.0:
            return variation_noise
        return self._slerp(base_noise, variation_noise, settings.strength)

    @staticmethod
    def _validate_noise(noise: torch.Tensor) -> None:
        """Require the dense floating batched tensor used by Comfy samplers."""

        if not isinstance(noise, torch.Tensor):
            raise TypeError("Seed variation requires sampler noise as a torch.Tensor.")
        if not noise.is_floating_point():
            raise TypeError("Seed variation requires floating-point sampler noise.")
        if noise.layout is not torch.strided:
            raise TypeError("Seed variation requires dense strided sampler noise.")
        if noise.ndim < 2:
            raise ValueError("Seed variation requires batched sampler noise.")

    @staticmethod
    def _is_disabled_noise(noise: torch.Tensor) -> bool:
        """Preserve an explicit all-zero noise request from the sampling node."""

        return noise.numel() == 0 or not bool(torch.count_nonzero(noise).item())

    @staticmethod
    def _variation_noise_like(noise: torch.Tensor, seed: int) -> torch.Tensor:
        """Create Comfy-compatible seeded CPU noise and move it to the base tensor."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        variation = torch.randn(
            tuple(noise.shape),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        return variation.to(device=noise.device, dtype=noise.dtype)

    @staticmethod
    def _slerp(
        base_noise: torch.Tensor,
        variation_noise: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Spherically interpolate each batch item with stable linear fallbacks."""

        batch_size = int(base_noise.shape[0])
        base_flat = base_noise.to(dtype=torch.float32).reshape(batch_size, -1)
        variation_flat = variation_noise.to(dtype=torch.float32).reshape(
            batch_size,
            -1,
        )
        base_norm = torch.linalg.vector_norm(base_flat, dim=1, keepdim=True)
        variation_norm = torch.linalg.vector_norm(
            variation_flat,
            dim=1,
            keepdim=True,
        )
        valid_norms = (base_norm > _SLERP_EPSILON) & (variation_norm > _SLERP_EPSILON)
        base_unit = base_flat / base_norm.clamp_min(_SLERP_EPSILON)
        variation_unit = variation_flat / variation_norm.clamp_min(_SLERP_EPSILON)
        cosine = (base_unit * variation_unit).sum(dim=1, keepdim=True)
        cosine = cosine.clamp(min=-1.0, max=1.0)
        angle = torch.acos(cosine)
        sine = torch.sin(angle)
        stable_angle = sine.abs() > _SLERP_EPSILON

        base_weight = torch.sin((1.0 - strength) * angle) / sine.clamp_min(
            _SLERP_EPSILON
        )
        variation_weight = torch.sin(strength * angle) / sine.clamp_min(_SLERP_EPSILON)
        spherical = base_weight * base_flat + variation_weight * variation_flat
        linear = torch.lerp(base_flat, variation_flat, strength)
        mixed = torch.where(valid_norms & stable_angle, spherical, linear)
        return mixed.reshape_as(base_noise).to(dtype=base_noise.dtype)


@dataclass(frozen=True, slots=True)
class SeedVariationOuterSampleWrapper:
    """Replace only the initial sampler noise before Comfy prepares the model."""

    settings: SeedVariationSettings
    interpolator: SeedVariationNoiseInterpolator = field(
        default_factory=SeedVariationNoiseInterpolator
    )

    def __call__(
        self,
        executor: Callable[..., object],
        noise: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward sampling with interpolated noise and every other argument intact."""

        varied_noise = self.interpolator.interpolate(noise, self.settings)
        return executor(varied_noise, *args, **kwargs)


class SeedVariationModelPatchBackend:
    """Derive a MODEL carrying one namespaced outer-sample wrapper."""

    def derive(self, model: object, settings: SeedVariationSettings) -> object:
        """Return the source for a no-op or a lifecycle-safe patched MODEL."""

        if settings.strength == 0.0:
            return model
        wrapper = SeedVariationOuterSampleWrapper(settings)
        return PATCHER_LIFECYCLE.derive_model(
            model,
            (
                ModelKeyedWrapperMutation(
                    wrapper_type=WrappersMP.OUTER_SAMPLE,
                    key=SEED_VARIATION_WRAPPER_KEY,
                    wrapper=wrapper,
                ),
            ),
            operation="seed variation MODEL patch",
        )


SEED_VARIATION_MODEL_PATCH_BACKEND = SeedVariationModelPatchBackend()
