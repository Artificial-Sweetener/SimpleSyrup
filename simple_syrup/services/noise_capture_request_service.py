"""Resolve upstream capture requests from the receiving sampler's live settings."""

from __future__ import annotations

import json
from typing import Any

import torch

from ..domain.sampler_options import SamplerOptions
from ..runtime.comfy_latent_normalization import COMFY_LATENT_NORMALIZER
from ..runtime.noise_inversion import validate_inversion_target
from ..runtime.sampling_schedulers import SchedulerView, calculate_sigmas


class NoiseCaptureRequestService:
    """Evaluate schedule planning before the originating sampler can execute."""

    def coordinates(
        self,
        *,
        model: Any,
        options: SamplerOptions,
        steps: int,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        source_latent: dict[str, Any],
        geometry_json: str,
        previous: tuple[float, ...] = (),
    ) -> tuple[float, ...]:
        """Calculate the same model-view schedule as the receiving sampling route."""
        if not isinstance(options, SamplerOptions):
            raise TypeError("Capture planning requires SimpleSyrup sampler options.")
        if options.noise_reuse is None:
            return previous
        samples = source_latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError("Capture planning requires source latent geometry.")
        context, tiling = options.contextual_diffusion, options.tiling
        if context is not None:
            view = SchedulerView(context.context_size, context.context_size)
        elif tiling is not None:
            view = SchedulerView(tiling.width, tiling.height)
        elif scheduler == "Flux2":
            samples = COMFY_LATENT_NORMALIZER.normalize(
                model=model,
                samples=samples,
                spatial_downscale_ratio=source_latent.get("downscale_ratio_spacial"),
            )
            view = self._view(samples, geometry_json)
        else:
            view = SchedulerView.from_tensor(samples)
        sigmas = calculate_sigmas(
            model, scheduler, sampler_name, steps, denoise, view=view
        )
        target = validate_inversion_target(model, sigmas)
        coordinate = options.noise_reuse.capture_coordinate(target)
        return tuple(sorted({*previous, coordinate}, reverse=True))

    def _view(self, samples: torch.Tensor, geometry_json: str) -> SchedulerView:
        """Resolve only validated native sizing operations for resolution-
        dependent schedules.
        """
        operations = json.loads(geometry_json)
        if not isinstance(operations, list):
            raise ValueError("Noise capture geometry must be an operation list.")
        height, width = int(samples.shape[-2]), int(samples.shape[-1])
        for operation in operations:
            if isinstance(operation, dict) and set(operation) == {"pixels"}:
                pixels = operation["pixels"]
                if (
                    not isinstance(pixels, list)
                    or len(pixels) != 2
                    or not all(type(value) is int and value >= 0 for value in pixels)
                ):
                    raise ValueError(
                        "Resolution-dependent noise capture requires literal "
                        "non-negative pixel dimensions."
                    )
                pixel_height, pixel_width = pixels
                if pixel_height == 0 and pixel_width == 0:
                    continue
                if pixel_width == 0:
                    pixel_height = max(64, pixel_height)
                    pixel_width = max(64, round(width * pixel_height / height))
                elif pixel_height == 0:
                    pixel_width = max(64, pixel_width)
                    pixel_height = max(64, round(height * pixel_width / width))
                else:
                    pixel_width, pixel_height = (
                        max(64, pixel_width),
                        max(64, pixel_height),
                    )
                height, width = pixel_height // 8, pixel_width // 8
                continue
            if not isinstance(operation, dict) or set(operation) != {"scale"}:
                raise ValueError(
                    "Noise capture geometry contains an unsupported operation."
                )
            scale = operation["scale"]
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not 0.01 <= scale <= 8
            ):
                raise ValueError(
                    "Noise capture geometry scale must be between 0.01 and 8."
                )
            height, width = round(height * scale), round(width * scale)
        return SchedulerView(width, height)
