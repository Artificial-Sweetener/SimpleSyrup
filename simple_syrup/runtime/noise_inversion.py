# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive source-dependent sampling noise with measured, uncached inversion stages."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TypeAlias

import torch

from ..domain.inversion_solver import (
    InversionSolverEvidence,
    integrate_inversion,
    lift_inversion_displacement,
)
from ..domain.noise_inversion import InversionMethod, NoiseInversionOptions
from ..shared.logging import get_logger
from .spatial_tensor_projection import resize_spatial_tensor
from .tiled_sampling_validation import validate_tensor_shape

LOGGER = get_logger(__name__)
INVERSION_START_SIGMA = 0.0001
InversionModelFactory: TypeAlias = Callable[[torch.Tensor], Any]


@dataclass(frozen=True, slots=True)
class InversionStageMeasurement:
    """Describe actual work and elapsed time for one inversion stage."""

    name: str
    seconds: float
    latent_shape: tuple[int, ...]
    steps: int
    evaluations: int


@dataclass(frozen=True, slots=True)
class NoiseInversionResult:
    """Return inferred noise with paid inversion cost and reconstruction evidence."""

    noise: torch.Tensor
    seconds: float
    stages: tuple[InversionStageMeasurement, ...]
    reconstruction_max_error: float


def _clock(device: torch.device) -> float:
    """Measure completed CUDA work rather than asynchronous kernel submission."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _source_in_model_space(model: Any, latent: torch.Tensor) -> torch.Tensor:
    """Narrow host latent processing before arithmetic or model execution."""
    source = model.model.process_latent_in(latent)
    if not isinstance(source, torch.Tensor) or source.shape != latent.shape:
        raise ValueError(
            "Model latent processing must preserve the inversion source shape."
        )
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
        raise ValueError(
            "Model latent processing must produce finite floating-point values."
        )
    return source.detach().float().cpu()


def validate_inversion_target(model: Any, sigmas: torch.Tensor) -> float:
    """Reject unsupported scaling and singular targets before model execution."""
    sampling_types = import_module("comfy.model_sampling")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(
        sampling, (sampling_types.CONST, sampling_types.EPS)
    ) or isinstance(
        sampling, (sampling_types.IMG_TO_IMG, sampling_types.IMG_TO_IMG_FLOW)
    ):
        raise ValueError(
            "Noise inversion requires a flow or EPS-compatible image model."
        )
    if sigmas.ndim != 1 or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError("Noise inversion requires a finite forward sampling schedule.")
    target = float(sigmas[0])
    if target <= INVERSION_START_SIGMA:
        raise ValueError(
            "Noise inversion requires a positive partial-denoise start sigma."
        )
    if isinstance(sampling, sampling_types.CONST) and target >= 0.9999:
        raise ValueError(
            "Flow noise inversion requires denoise below the full-noise endpoint."
        )
    if isinstance(sampling, sampling_types.EPS):
        maximum = float(sampling.sigma_max)
        if target > maximum or math.isclose(target, maximum, rel_tol=1e-5):
            raise ValueError(
                "Noise inversion requires partial denoise, below the model's "
                "maximum sigma."
            )
    return target


def invert_sampling_noise(
    *,
    model: Any,
    latent: torch.Tensor,
    forward_sigmas: torch.Tensor,
    positive: Any,
    negative: Any,
    cfg: float,
    seed: int | None,
    options: NoiseInversionOptions,
    model_factory: InversionModelFactory | None = None,
    noise_mask: Any = None,
) -> NoiseInversionResult:
    """Invert a source latent through Comfy's actual CFG or positive-only guider.

    A spatial sampler supplies a factory that replans each resolution from its
    unwrapped prepared model. The final noise recreates the inferred endpoint
    under the model's own affine noise scaling; no inference cache is used.
    """
    from .guided_sampling import sample_with_optional_negative

    if not isinstance(options, NoiseInversionOptions):
        raise TypeError("Noise inversion requires validated NoiseInversionOptions.")
    validate_tensor_shape(latent, sampler_label="Noise Inversion")
    if not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
        raise ValueError(
            "Noise inversion source must contain finite floating-point values."
        )
    target = validate_inversion_target(model, forward_sigmas)
    coarse_target = target * options.coarse_target_fraction
    if coarse_target <= INVERSION_START_SIGMA:
        raise ValueError(
            "Noise inversion transition must exceed the initial inversion sigma."
        )
    device = torch.device(model.load_device)
    started = _clock(device)
    sampling = model.get_model_object("model_sampling")
    source = _source_in_model_space(model, latent)
    phases: list[InversionStageMeasurement] = []
    comfy_sample = import_module("comfy.sample")
    comfy_samplers = import_module("comfy.samplers")

    def stage(
        stage_latent: torch.Tensor,
        begin: float,
        end: float,
        count: int,
        initial: torch.Tensor | None,
        name: str,
        method: InversionMethod,
    ) -> torch.Tensor:
        """Capture the model-space endpoint before Comfy converts output latents."""
        stage_started = _clock(device)
        stage_model = (
            model_factory(stage_latent) if model_factory is not None else model
        )
        schedule = torch.linspace(begin, end, count + 1, device=device)
        evidence = InversionSolverEvidence()
        endpoints: list[torch.Tensor] = []

        def invert(
            model_fn: Any,
            state: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: dict[str, Any],
            callback: Any,
            disable: bool,
        ) -> torch.Tensor:
            """Use Comfy's denoised predictions as the inversion velocity field."""
            if initial is not None:
                state = initial.to(state)

            def evaluate(
                x: torch.Tensor, sigma: torch.Tensor, index: int
            ) -> torch.Tensor:
                """Narrow the dynamic Comfy model result before numeric integration."""
                prediction = model_fn(
                    x, sigma * x.new_ones((x.shape[0],)), **extra_args
                )
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError(
                        "Noise inversion model must return tensor predictions."
                    )
                return (x - prediction) / sigma

            endpoint = integrate_inversion(
                state, sigmas, evaluate, method=method, evidence=evidence
            )
            endpoints.append(endpoint.detach().float().cpu())
            return endpoint

        sample_with_optional_negative(
            comfy_sample=comfy_sample,
            model=stage_model,
            noise=torch.zeros_like(stage_latent),
            cfg=cfg,
            sampler=comfy_samplers.KSAMPLER(invert),
            sigmas=schedule,
            positive=positive,
            negative=negative,
            latent_image=stage_latent,
            noise_mask=noise_mask,
            seed=seed,
            disable_pbar=True,
        )
        if len(endpoints) != 1:
            raise RuntimeError(
                "Noise inversion must produce exactly one endpoint per stage."
            )
        phases.append(
            InversionStageMeasurement(
                name,
                _clock(device) - stage_started,
                tuple(stage_latent.shape),
                count,
                evidence.evaluations,
            )
        )
        return endpoints[0]

    if options.resolution_scale == 1:
        endpoint = stage(
            latent,
            INVERSION_START_SIGMA,
            target,
            options.steps,
            None,
            "full",
            options.method,
        )
    else:
        height, width = options.coarse_shape(
            int(latent.shape[-2]), int(latent.shape[-1])
        )
        coarse = resize_spatial_tensor(latent, height=height, width=width, mode="area")
        coarse_source = _source_in_model_space(model, coarse)
        coarse_endpoint = stage(
            coarse,
            INVERSION_START_SIGMA,
            coarse_target,
            options.steps,
            None,
            "coarse",
            options.method,
        )
        endpoint = lift_inversion_displacement(
            source,
            coarse_source,
            coarse_endpoint,
            resize=lambda x, h, w: resize_spatial_tensor(
                x, height=h, width=w, mode="bilinear"
            ),
        )
        if options.finishing_steps:
            endpoint = stage(
                latent,
                coarse_target,
                target,
                options.finishing_steps,
                endpoint,
                "full_finish",
                options.finishing_method,
            )

    sigma = torch.tensor(target)
    zero = torch.zeros_like(source)
    base = sampling.noise_scaling(sigma, zero.clone(), source, max_denoise=False)
    amplitude = sampling.noise_scaling(
        sigma, torch.ones_like(source), zero, max_denoise=False
    )
    if not isinstance(base, torch.Tensor) or not isinstance(amplitude, torch.Tensor):
        raise TypeError("Model noise scaling must return tensors.")
    if not bool(torch.isfinite(amplitude).all()) or bool(torch.any(amplitude == 0)):
        raise ValueError("Model noise scaling is not invertible at the target sigma.")
    noise = (endpoint - base) / amplitude
    if not bool(torch.isfinite(noise).all()):
        raise FloatingPointError("Noise inversion produced non-finite sampling noise.")
    reconstructed = sampling.noise_scaling(
        sigma, noise.clone(), source, max_denoise=False
    )
    if (
        not isinstance(reconstructed, torch.Tensor)
        or reconstructed.shape != endpoint.shape
    ):
        raise ValueError(
            "Model noise scaling must preserve the inversion endpoint shape."
        )
    if not torch.allclose(reconstructed, endpoint, atol=1e-5, rtol=1e-5):
        raise ValueError(
            "Model noise scaling cannot reconstruct the inversion endpoint."
        )
    error = float((reconstructed - endpoint).abs().max())
    elapsed = _clock(device) - started
    LOGGER.info(
        "Noise inversion completed in %.3f seconds",
        elapsed,
        extra={
            "operation": "noise_inversion",
            "method": options.method,
            "resolution_scale": options.resolution_scale,
            "steps": options.steps,
            "finishing_steps": options.finishing_steps,
            "inversion_seconds": elapsed,
            "inversion_stages": [
                {
                    "name": phase.name,
                    "seconds": phase.seconds,
                    "latent_shape": phase.latent_shape,
                    "steps": phase.steps,
                    "evaluations": phase.evaluations,
                }
                for phase in phases
            ],
            "evaluations": sum(phase.evaluations for phase in phases),
            "endpoint_reconstruction_max_error": error,
        },
    )
    return NoiseInversionResult(noise, elapsed, tuple(phases), error)
