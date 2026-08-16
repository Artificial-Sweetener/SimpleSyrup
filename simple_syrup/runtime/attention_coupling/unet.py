# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive a standard-UNet MODEL with regional attention ownership."""

from __future__ import annotations

from dataclasses import dataclass

from ..model_attention_patch_mutations import ModelAttn2PatchesMutation
from ..patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation
from ..regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
)
from ..regional_lora.standard_unet_variant_runtime import (
    StandardUnetVariantRuntimeMutation,
)
from ..regional_lora.standard_unet_variant_template import (
    STANDARD_UNET_VARIANT_TEMPLATE_CACHE,
)
from .unet_attention_context_wrapper import unet_attention_context_wrapper_mutation
from .unet_attention_phase_session import StandardUnetAttentionPhaseSession
from .unet_attention_state import StandardUnetAttentionState
from .unet_attn2_execution_resolver import StandardUnetAttn2ExecutionResolver
from .unet_attn2_patch import UnetAttn2PatchPair


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionModel:
    """Return one derived MODEL with its immutable attention execution."""

    model: object
    state: StandardUnetAttentionState


class StandardUnetAttentionBackend:
    """Install regional cross-attention on a collision-safe model clone."""

    def derive(
        self,
        *,
        model: object,
        state: StandardUnetAttentionState,
        admission: StandardUnetNativeLoraAdmission,
    ) -> StandardUnetAttentionModel:
        """Return a direct MODEL child containing only the paired UNet patches."""

        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet backend requires attention state.")
        if not isinstance(admission, StandardUnetNativeLoraAdmission):
            raise TypeError("Standard UNet backend requires native admission.")
        if admission.adaptation.plan != state.plan.lora_plan:
            raise ValueError(
                "Standard UNet admission and processed conditioning must share "
                "the same regional LoRA plan."
            )
        attention_phase = StandardUnetAttentionPhaseSession()
        template = (
            STANDARD_UNET_VARIANT_TEMPLATE_CACHE.resolve(model, admission)
            if admission.adaptation.plan.adapters
            else None
        )
        variant_mutations = (
            (
                StandardUnetVariantRuntimeMutation(
                    state,
                    admission,
                    attention_phase,
                    template,
                ),
            )
            if template is not None
            else ()
        )
        derivation_source = (
            template.bind_request(model) if template is not None else model
        )
        attention_mutations: tuple[ModelMutation, ...] = ()
        if template is None:
            patches = UnetAttn2PatchPair(
                StandardUnetAttn2ExecutionResolver(state),
            )
            attention_mutations = (
                ModelAttn2PatchesMutation(
                    patches.input_patch,
                    patches.output_patch,
                ),
            )
        derived = PATCHER_LIFECYCLE.derive_model(
            derivation_source,
            (
                unet_attention_context_wrapper_mutation(
                    state,
                    attention_phase,
                ),
                *attention_mutations,
                *variant_mutations,
            ),
            operation="standard UNet Attention Coupling",
        )
        return StandardUnetAttentionModel(derived, state)


STANDARD_UNET_ATTENTION_BACKEND = StandardUnetAttentionBackend()
