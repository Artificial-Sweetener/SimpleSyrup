# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Supply model-local NegPiP attention hooks for Comfy's earlier Krea boundary."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from inspect import signature
from typing import Any, cast

import torch
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.krea2.model import Attention
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.model_patcher import ModelPatcher
from einops import rearrange

from ..model_patcher_mutations import ModelCallableObjectPatchMutation
from .krea2 import TRANSFORMER_MASK_KEY

LOGGER = logging.getLogger(__name__)


def krea2_host_mutations(
    model: ModelPatcher,
) -> tuple[ModelCallableObjectPatchMutation, ...]:
    """Adapt only the known five-argument host; retain native reference-capable Krea."""
    diffusion = model.get_model_object("diffusion_model")
    parameters = tuple(signature(diffusion._forward).parameters)
    if "ref_latents" in parameters:
        return ()
    if parameters != (
        "x",
        "timesteps",
        "context",
        "attention_mask",
        "transformer_options",
        "kwargs",
    ):
        raise ValueError(
            f"Krea NegPiP does not support model signature {parameters!r}."
        )
    mutations: list[ModelCallableObjectPatchMutation] = []
    for index, block in enumerate(diffusion.blocks):
        attention = block.attn
        if not isinstance(attention, Attention):
            raise TypeError(f"Krea NegPiP requires host Attention at block {index}.")
        mutations.append(
            ModelCallableObjectPatchMutation(
                f"diffusion_model.blocks.{index}.attn.forward",
                Krea2HostAttention(attention, index, len(diffusion.blocks)),
            )
        )
    if not mutations:
        raise ValueError("Krea NegPiP requires at least one joint attention block.")
    LOGGER.info(
        "Krea NegPiP installed model-local host attention hooks",
        extra={"blocks": len(mutations)},
    )
    return tuple(mutations)


@dataclass(frozen=True)
class Krea2HostAttention:
    """Preserve host attention math while exposing its missing pre-RoPE patch point.

    Comfy 0.28's public model boundary lacks attention callbacks. Object patches
    scope this adapter to the derived MODEL and Comfy restores them on unload.
    Text-fusion attention stays untouched; only joint text/image blocks use it.
    """

    attention: Any  # Comfy Attention exposes dynamically constructed linear modules.
    block_index: int
    total_blocks: int

    def __call__(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run host QKV projections and attention with local patch metadata."""
        options = {} if transformer_options is None else transformer_options.copy()
        multiplier = options.get(TRANSFORMER_MASK_KEY)
        if multiplier is None:
            return cast(
                torch.Tensor,
                type(self.attention).forward(
                    self.attention,
                    x,
                    freqs,
                    mask,
                    transformer_options=options,
                ),
            )
        if not isinstance(multiplier, torch.Tensor) or multiplier.ndim != 3:
            raise ValueError(
                "Krea NegPiP host attention requires a processed token sign mask."
            )
        options.update(
            block_index=self.block_index,
            total_blocks=self.total_blocks,
            block_type="single",
            img_slice=[multiplier.shape[1], x.shape[1]],
        )
        attention = self.attention
        q, k, v, gate = (
            attention.wq(x),
            attention.wk(x),
            attention.wv(x),
            attention.gate(x),
        )
        q = rearrange(q, "B L (H D) -> B H L D", H=attention.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=attention.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=attention.kvheads)
        q, k = attention.qknorm(q, k)
        for patch in options.get("patches", {}).get("attn1_patch", []):
            result = patch(
                q, k, v, pe=freqs, attn_mask=mask, extra_options=options.copy()
            )
            q, k, v = result.get("q", q), result.get("k", k), result.get("v", v)
            freqs, mask = result.get("pe", freqs), result.get("attn_mask", mask)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if attention.kvheads != attention.heads:
            repeats = attention.heads // attention.kvheads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        out = optimized_attention_masked(
            q,
            k,
            v,
            attention.heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=options,
        )
        for patch in options.get("patches", {}).get("attn1_output_patch", []):
            out = patch(out, options.copy())
        return cast(torch.Tensor, attention.wo(out * torch.nn.functional.sigmoid(gate)))
