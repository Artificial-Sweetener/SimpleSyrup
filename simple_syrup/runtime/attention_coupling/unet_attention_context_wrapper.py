# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish active regional contexts around each standard-UNet call."""

from __future__ import annotations

import torch

from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..diffusion_wrapper_invocation import DIFFUSION_WRAPPER_INVOCATION_VALIDATOR
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from ..regional_attention_model_call import RegionalAttentionModelCallResolver
from .standard_unet_model_output_validation import (
    STANDARD_UNET_MODEL_OUTPUT_VALIDATOR,
    StandardUnetModelOutputValidator,
)
from .unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)
from .unet_attention_state import StandardUnetAttentionState
from .unet_model_call_resolver import STANDARD_UNET_MODEL_CALL_RESOLVER

UNET_ATTENTION_CONTEXT_WRAPPER_KEY = "simple_syrup.unet_regional_attention_contexts"


class StandardUnetAttentionContextDiffusionWrapper:
    """Adapt the installed UNet call contract to shared active context state."""

    def __init__(
        self,
        state: StandardUnetAttentionState,
        attention_phase: StandardUnetAttentionPhaseSession,
        *,
        model_call_resolver: RegionalAttentionModelCallResolver = (
            STANDARD_UNET_MODEL_CALL_RESOLVER
        ),
        output_validator: StandardUnetModelOutputValidator = (
            STANDARD_UNET_MODEL_OUTPUT_VALIDATOR
        ),
    ) -> None:
        """Retain the shared plan and model-call authorities."""

        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet context wrapper requires attention state.")
        if not isinstance(attention_phase, StandardUnetAttentionPhaseSession):
            raise TypeError("Standard UNet context wrapper requires phase state.")
        if not isinstance(model_call_resolver, RegionalAttentionModelCallResolver):
            raise TypeError(
                "Standard UNet context wrapper requires a model-call resolver."
            )
        self._state = state
        self._attention_phase = attention_phase
        self._model_call_resolver = model_call_resolver
        if not isinstance(output_validator, StandardUnetModelOutputValidator):
            raise TypeError("Standard UNet output validator has an invalid type.")
        self._output_validator = output_validator

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Resolve and publish contexts for exactly one nested UNet execution."""

        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("Standard UNet context wrapper requires model input.")
        if len(args) < 3 or not isinstance(args[2], torch.Tensor):
            raise TypeError(
                "Standard UNet context wrapper requires a tensor third positional "
                "base context."
            )
        if len(args) < 6 or not isinstance(args[5], dict):
            raise TypeError(
                "Standard UNet context wrapper requires dictionary sixth positional "
                "transformer options."
            )
        DIFFUSION_WRAPPER_INVOCATION_VALIDATOR.require_owned(
            executor,
            args[5],
            key=UNET_ATTENTION_CONTEXT_WRAPPER_KEY,
            wrapper=self,
        )
        contexts = self._model_call_resolver.resolve(
            self._state.plan,
            model_input=args[0],
            base_context=args[2],
            transformer_options=args[5],
        )
        forwarded_args = (*args[:2], contexts.base_context, *args[3:])
        with (
            self._attention_phase.activate(args[5]),
            self._state.execution_context.activate(contexts),
            self._state.resolution_cache.activate(),
        ):
            output = executor(*forwarded_args, **kwargs)
            return self._output_validator.validate(output, model_input=args[0])


def unet_attention_context_wrapper_mutation(
    state: StandardUnetAttentionState,
    attention_phase: StandardUnetAttentionPhaseSession,
) -> ModelDiffusionWrapperMutation:
    """Return the clone-local standard-UNet context wrapper mutation."""

    return ModelDiffusionWrapperMutation(
        UNET_ATTENTION_CONTEXT_WRAPPER_KEY,
        StandardUnetAttentionContextDiffusionWrapper(
            state,
            attention_phase,
        ),
    )
