# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive a standard-UNet MODEL with paired regional attn2 patches."""

from __future__ import annotations

from dataclasses import dataclass

import comfy.model_patcher
from comfy.patcher_extension import CallbacksMP

from ..model_patcher_mutations import (
    ModelAttn2PatchesMutation,
    ModelKeyedCallbackMutation,
)
from ..patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation
from ..regional_lora.operation_mutation_assembly import (
    REGIONAL_OPERATION_MUTATION_ASSEMBLER,
)
from ..regional_lora.standard_unet_operation_preparation import (
    StandardUnetOperationAdmission,
)
from ..regional_lora.standard_unet_operation_session import (
    StandardUnetRegionalOperationSession,
)
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
        admission: StandardUnetOperationAdmission,
    ) -> StandardUnetAttentionModel:
        """Return a direct MODEL child containing only the paired UNet patches."""

        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet backend requires attention state.")
        if not isinstance(admission, StandardUnetOperationAdmission):
            raise TypeError("Standard UNet backend requires operation admission.")
        if admission.adaptation.plan != state.plan.lora_plan:
            raise ValueError(
                "Standard UNet admission and processed conditioning must share "
                "the same regional LoRA plan."
            )
        base_model = getattr(model, "model", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None:
            raise TypeError("Standard UNet MODEL must expose model.diffusion_model.")
        operation_session: StandardUnetRegionalOperationSession | None = None
        operation_mutations: tuple[ModelMutation, ...] = ()
        schedule_mutations: tuple[ModelMutation, ...] = ()
        if admission.adaptation.plan.adapters:
            if (
                not isinstance(model, comfy.model_patcher.ModelPatcher)
                or admission.binding is None
                or admission.cache is None
            ):
                raise TypeError(
                    "Standard UNet regional operations require complete MODEL "
                    "admission."
                )
            operation_session = StandardUnetRegionalOperationSession(
                admission.adaptation.plan,
                state.plan.mask_bank,
                admission.module_roles,
            )
            operation_mutations = REGIONAL_OPERATION_MUTATION_ASSEMBLER.assemble(
                admission.binding,
                model=model,
                cache=admission.cache,
            )
            schedule_mutations = (
                ModelKeyedCallbackMutation(
                    CallbacksMP.ON_DETACH,
                    "simple_syrup.standard_unet_regional_operation_schedule",
                    operation_session.clear,
                ),
            )
        patches = UnetAttn2PatchPair(
            StandardUnetAttn2ExecutionResolver(state),
            operation_scope=operation_session,
        )
        derived = PATCHER_LIFECYCLE.derive_model(
            model,
            (
                unet_attention_context_wrapper_mutation(
                    diffusion_model,
                    state,
                    operation_session=operation_session,
                ),
                ModelAttn2PatchesMutation(
                    patches.input_patch,
                    patches.output_patch,
                ),
                *operation_mutations,
                *schedule_mutations,
            ),
            operation="standard UNet Attention Coupling",
        )
        return StandardUnetAttentionModel(derived, state)


STANDARD_UNET_ATTENTION_BACKEND = StandardUnetAttentionBackend()
