"""Prepare captured residuals for the shared native sampler starting-noise input."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.spectral_noise import lift_residual
from ..shared.logging import get_logger
from .native_noise_scaling import noise_coefficients, recover_residual
from .noise_inversion import InversionModelFactory, validate_inversion_target
from .noise_reuse_finishing import finish_reused_state
from .noise_reuse_scope import NoiseReuseExecution

LOGGER = get_logger(__name__)


def prepare_reused_noise(
    *,
    execution: NoiseReuseExecution,
    model: Any,
    fresh: torch.Tensor,
    latent: torch.Tensor,
    sigmas: torch.Tensor,
    positive: Any,
    negative: Any,
    cfg: float,
    seed: int | None,
    noise_mask: Any,
    model_factory: InversionModelFactory | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift source coefficients and return the expected actual model-space
    starting state.
    """
    if execution.consumed:
        raise ValueError(
            "Noise reuse supports one native sampling pass per "
            "receiver; use separate samplers for conditioning "
            "batches."
        )
    execution.consumed = True
    target = validate_inversion_target(model, sigmas)
    coordinate = execution.options.capture_coordinate(target)
    provenance = execution.provenance
    if type(model.model.latent_format).__name__ != provenance.latent_format:
        raise ValueError(
            "Noise reuse source and receiving model use different latent formats."
        )
    scaling = model.get_model_object("model_sampling")
    residual = provenance.residual_at(coordinate)
    lifted = lift_residual(residual, fresh.detach().float().cpu())
    clean = model.model.process_latent_in(latent).detach().float().cpu()
    if execution.options.finishing_steps:
        base, amplitude = noise_coefficients(scaling, clean, coordinate)
        endpoint = finish_reused_state(
            model=model,
            latent=latent,
            initial=base + amplitude * lifted,
            begin=coordinate,
            target=target,
            options=execution.options,
            positive=positive,
            negative=negative,
            cfg=cfg,
            seed=seed,
            noise_mask=noise_mask,
            model_factory=model_factory,
        )
        lifted = recover_residual(scaling, clean, endpoint, target)
    base, amplitude = noise_coefficients(scaling, clean, target)
    expected = base + amplitude * lifted
    LOGGER.info(
        "Captured sampling noise prepared",
        extra={
            "operation": "noise_reuse",
            "capture_sigma": coordinate,
            "target_sigma": target,
            "source_shape": tuple(residual.shape),
            "target_shape": tuple(lifted.shape),
            "finishing_steps": execution.options.finishing_steps,
        },
    )
    return lifted.to(fresh), expected
