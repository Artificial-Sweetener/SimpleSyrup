# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply PPM-compatible value-mask NegPiP behavior to Anima."""

# NegPiP behavior is adapted from ComfyUI-ppm and its credited predecessors.
# See third_party/manifest.toml and third_party/NOTICE.md.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from comfy import conds

WRAPPER_KEY = "ppm_negpip_anima"
CONDITION_MASK_KEY = "c_ppm_negpip_mask"
TRANSFORMER_MASK_KEY = "ppm_negpip_mask"


def anima_extra_conds_negpip_wrapper(
    previous_extra_conds: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Convert signed T5 weights into a model condition while preserving magnitude."""

    def wrapped_extra_conds(**kwargs: object) -> dict[str, object]:
        """Publish a sequence-aligned value multiplier for one conditioning."""

        weights = kwargs.get("t5xxl_weights")
        multiplier: torch.Tensor | None = None
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                raise TypeError("Anima NegPiP T5 weights must be a tensor.")
            magnitude = weights.abs()
            multiplier = (
                torch.where(
                    weights < 0.0,
                    weights.new_tensor(-1.0),
                    weights.new_tensor(1.0),
                )
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            if multiplier.shape[1] < 512:
                multiplier = torch.nn.functional.pad(
                    multiplier,
                    (0, 0, 0, 512 - multiplier.shape[1]),
                    value=1.0,
                )
            kwargs["t5xxl_weights"] = magnitude

        output = previous_extra_conds(**kwargs)
        if not isinstance(output, dict):
            raise TypeError("Anima extra conditions must be a dictionary.")
        if multiplier is not None:
            output[CONDITION_MASK_KEY] = conds.CONDRegular(multiplier)
        return output

    return wrapped_extra_conds


def anima_diffusion_negpip_wrapper(
    executor: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """Move the processed Anima multiplier into isolated transformer options."""

    if len(args) < 3 or not isinstance(args[2], torch.Tensor):
        raise TypeError("Anima NegPiP wrapper requires tensor conditioning context.")
    context = args[2]
    transformer_options = kwargs.get("transformer_options", {})
    if not isinstance(transformer_options, dict):
        raise TypeError("Anima transformer options must be a dictionary.")
    prepared = transformer_options.copy()
    multiplier = kwargs.get(CONDITION_MASK_KEY)
    if multiplier is not None:
        if not isinstance(multiplier, torch.Tensor):
            raise TypeError("Anima NegPiP multiplier must be a tensor.")
        prepared[TRANSFORMER_MASK_KEY] = multiplier.to(context)
    kwargs["transformer_options"] = prepared
    return executor(*args, **kwargs)


def anima_attn2_negpip(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pe: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Apply the signed multiplier only to Anima cross-attention values."""

    multiplier = (
        None if extra_options is None else extra_options.get(TRANSFORMER_MASK_KEY)
    )
    if multiplier is not None and not isinstance(multiplier, torch.Tensor):
        raise TypeError("Anima NegPiP attention multiplier must be a tensor.")
    return {
        "q": query,
        "k": key,
        "v": value if multiplier is None else value * multiplier,
        "pe": pe,
        "attn_mask": attn_mask,
    }
