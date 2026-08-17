# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify transparent standard-UNet cold sampling instrumentation."""

from __future__ import annotations

import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_cold_sampling import (
    StandardUnetColdSamplingDiagnosticsMutation,
)


def test_mutation_installs_one_keyed_sampler_wrapper() -> None:
    """Keep sampling attribution distinct from diffusion execution wrappers."""

    patcher = ModelPatcher(nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))

    StandardUnetColdSamplingDiagnosticsMutation().apply(patcher)

    wrappers = patcher.get_wrappers(
        WrappersMP.SAMPLER_SAMPLE,
        "simple_syrup.standard_unet_cold_sampling",
    )
    assert len(wrappers) == 1


def test_wrapper_forwards_exact_arguments_and_result() -> None:
    """Make disabled diagnostics observational for ordinary execution."""

    received: tuple[object, ...] = ()

    def executor(*args: object, **kwargs: object) -> str:
        nonlocal received
        received = (*args, kwargs)
        return "sampled"

    mutation = StandardUnetColdSamplingDiagnosticsMutation()
    sigmas = torch.tensor([1.0, 0.0])
    noise = torch.zeros((1, 4, 8, 8))

    result = mutation.measure_sampling(
        executor,
        "guider",
        sigmas,
        {"seed": 1},
        "callback",
        noise,
        "latent",
        disable_pbar=True,
    )

    assert result == "sampled"
    assert received == (
        "guider",
        sigmas,
        {"seed": 1},
        "callback",
        noise,
        "latent",
        {"disable_pbar": True},
    )
