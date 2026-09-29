# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve pinned RES4LYF solver methods within ComfyUI's sampler boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import torch

from .res4lyf_sampler_names import RES4LYF_SAMPLER_NAMES
from .sampling_samplers import SamplerObject


def resolve_res4lyf_sampler(sampler_name: str) -> SamplerObject:
    """Bind an upstream method name to the pinned RES4LYF solver."""

    if sampler_name not in RES4LYF_SAMPLER_NAMES:
        raise ValueError(f"Unsupported RES4LYF sampler '{sampler_name}'.")

    method = sampler_name.rsplit("/", 1)[-1]
    implicit = sampler_name.startswith(("fully_implicit/", "diag_implicit/"))
    options = {
        "rk_type": "euler" if implicit else method,
        "implicit_sampler_name": method if implicit else "use_explicit",
        "implicit_type": "bongmath",
        "implicit_type_substeps": "bongmath",
        "bongmath": sampler_name != "linear/rk5_7s",
    }
    comfy_samplers = import_module("comfy.samplers")
    return cast(
        SamplerObject,
        comfy_samplers.KSAMPLER(_sample_res4lyf, extra_options=options),
    )


def _sample_res4lyf(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    extra_args: dict[str, Any],
    callback: Any,
    disable: bool,
    rk_type: str,
    implicit_sampler_name: str,
    implicit_type: str,
    implicit_type_substeps: str,
    bongmath: bool,
) -> torch.Tensor:
    """Pass ComfyUI's seed to the same SDE stream used by RES4LYF's node."""

    seed = extra_args.get("seed")
    if not isinstance(seed, int):
        raise ValueError("RES4LYF sampler requires an integer sampling seed.")
    solver = import_module(
        "simple_syrup.third_party.res4lyf_runtime.beta.rk_sampler_beta"
    )
    samples = cast(
        torch.Tensor,
        solver.sample_rk_beta(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            rk_type=rk_type,
            implicit_sampler_name=implicit_sampler_name,
            implicit_type=implicit_type,
            implicit_type_substeps=implicit_type_substeps,
            BONGMATH=bongmath,
            noise_seed=seed + 1,
        ),
    )
    if not torch.isfinite(samples).all():
        raise ValueError(
            f"RES4LYF sampler '{rk_type}' produced non-finite latent values. "
            "Try another sampler or scheduler for this model."
        )
    return samples
