# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive a standard-UNet MODEL with paired regional attn2 patches."""

from __future__ import annotations

from dataclasses import dataclass

from ..model_patcher_mutations import ModelAttn2PatchesMutation
from ..patcher_lifecycle import PATCHER_LIFECYCLE
from .unet_attention_context_wrapper import unet_attention_context_wrapper_mutation
from .unet_attention_state import StandardUnetAttentionState
from .unet_attn2_execution_resolver import StandardUnetAttn2ExecutionResolver
from .unet_attn2_patch import UnetAttn2PatchPair


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionModel:
    """Return one derived MODEL with its immutable attention execution."""

    model: object
    state: StandardUnetAttentionState


class StandardUnetAttentionBackend:
    """Install one paired attn2 callback owner on a collision-safe clone."""

    def derive(
        self,
        *,
        model: object,
        state: StandardUnetAttentionState,
    ) -> StandardUnetAttentionModel:
        """Return a direct MODEL child containing only the paired UNet patches."""

        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet backend requires attention state.")
        base_model = getattr(model, "model", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None:
            raise TypeError("Standard UNet MODEL must expose model.diffusion_model.")
        patches = UnetAttn2PatchPair(StandardUnetAttn2ExecutionResolver(state))
        derived = PATCHER_LIFECYCLE.derive_model(
            model,
            (
                unet_attention_context_wrapper_mutation(diffusion_model, state),
                ModelAttn2PatchesMutation(
                    patches.input_patch,
                    patches.output_patch,
                ),
            ),
            operation="standard UNet Attention Coupling",
        )
        return StandardUnetAttentionModel(derived, state)


STANDARD_UNET_ATTENTION_BACKEND = StandardUnetAttentionBackend()
