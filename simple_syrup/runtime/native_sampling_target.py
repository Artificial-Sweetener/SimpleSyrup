# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate shared model scaling and partial-denoise receiver coordinates."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import torch

MINIMUM_PARTIAL_SIGMA = 0.0001


def validate_partial_sampling_target(
    model: Any, sigmas: torch.Tensor, *, operation: str
) -> float:
    """Reject unsupported scaling and singular targets before model execution."""
    sampling_types = import_module("comfy.model_sampling")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(
        sampling, (sampling_types.CONST, sampling_types.EPS)
    ) or isinstance(
        sampling, (sampling_types.IMG_TO_IMG, sampling_types.IMG_TO_IMG_FLOW)
    ):
        raise ValueError(f"{operation} requires a flow or EPS-compatible image model.")
    if sigmas.ndim != 1 or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError(f"{operation} requires a finite forward sampling schedule.")
    target = float(sigmas[0])
    if target <= MINIMUM_PARTIAL_SIGMA:
        raise ValueError(
            f"{operation} requires a positive partial-denoise start sigma."
        )
    if isinstance(sampling, sampling_types.CONST) and target >= 0.9999:
        raise ValueError(
            f"Flow {operation.lower()} requires denoise below the full-noise endpoint."
        )
    if isinstance(sampling, sampling_types.EPS):
        maximum = float(sampling.sigma_max)
        if target > maximum or math.isclose(target, maximum, rel_tol=1e-5):
            raise ValueError(
                f"{operation} requires partial denoise, below the model's "
                "maximum sigma."
            )
    return target
