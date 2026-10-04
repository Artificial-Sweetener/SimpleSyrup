"""Advance an already reconstructed state through full-resolution inversion
finishing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from ..domain.inversion_solver import InversionSolverEvidence, integrate_inversion
from ..domain.noise_reuse import NoiseReuseOptions
from ..shared.logging import get_logger
from .noise_inversion import InversionModelFactory
from .noise_reuse_scope import noise_reuse_scope

LOGGER = get_logger(__name__)


def finish_reused_state(
    *,
    model: Any,
    latent: torch.Tensor,
    initial: torch.Tensor,
    begin: float,
    target: float,
    options: NoiseReuseOptions,
    positive: Any,
    negative: Any,
    cfg: float,
    seed: int | None,
    noise_mask: Any,
    model_factory: InversionModelFactory | None,
) -> torch.Tensor:
    """Run only the requested finishing evaluations, using native CFG and
    spatial policy.
    """
    from .guided_sampling import sample_with_optional_negative

    if options.finishing_steps < 1 or begin >= target:
        raise ValueError(
            "Finishing requires positive steps and a start below the "
            "sampler coordinate."
        )
    device = torch.device(model.load_device)
    schedule = torch.linspace(begin, target, options.finishing_steps + 1, device=device)
    evidence = InversionSolverEvidence()
    endpoints: list[torch.Tensor] = []

    def integrate(
        model_fn: Any,
        state: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict[str, Any],
        callback: Any,
        disable: bool,
    ) -> torch.Tensor:
        """Consume the captured initial state without adding a second layer of noise."""
        state = initial.to(state)

        def evaluate(x: torch.Tensor, sigma: torch.Tensor, index: int) -> torch.Tensor:
            """Use the production inversion velocity from native denoised
            predictions.
            """
            prediction = model_fn(x, sigma * x.new_ones((x.shape[0],)), **extra_args)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("Finishing model must return tensor predictions.")
            return (x - prediction) / sigma

        endpoint = integrate_inversion(
            state, sigmas, evaluate, method=options.finishing_method, evidence=evidence
        )
        endpoints.append(endpoint.detach().float().cpu())
        return endpoint

    stage_model = model_factory(latent) if model_factory is not None else model
    with noise_reuse_scope(None):
        sample_with_optional_negative(
            comfy_sample=import_module("comfy.sample"),
            model=stage_model,
            noise=torch.zeros_like(latent),
            cfg=cfg,
            sampler=import_module("comfy.samplers").KSAMPLER(integrate),
            sigmas=schedule,
            positive=positive,
            negative=negative,
            latent_image=latent,
            noise_mask=noise_mask,
            seed=seed,
            disable_pbar=True,
        )
    if len(endpoints) != 1:
        raise RuntimeError("Finishing must produce exactly one native endpoint.")
    LOGGER.info(
        "Captured noise finishing completed",
        extra={
            "operation": "noise_reuse_finish",
            "steps": options.finishing_steps,
            "evaluations": evidence.evaluations,
            "begin_sigma": begin,
            "target_sigma": target,
            "coarse_inversion_evaluations": 0,
        },
    )
    return endpoints[0]
