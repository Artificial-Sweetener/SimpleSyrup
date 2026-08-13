# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish active regional contexts around each standard-UNet call."""

from __future__ import annotations

from weakref import ReferenceType, ref

import torch

from ..diffusion_wrapper_executor import DiffusionWrapperExecutor
from ..model_patcher_mutations import ModelDiffusionWrapperMutation
from ..regional_attention_model_call import (
    REGIONAL_ATTENTION_MODEL_CALL_RESOLVER,
    RegionalAttentionModelCallResolver,
)
from ..regional_lora.operation_invocation import (
    REGIONAL_OPERATION_INVOCATION_CONTEXT,
)
from ..regional_lora.standard_unet_operation_session import (
    StandardUnetRegionalOperationSession,
)
from .unet_attention_state import StandardUnetAttentionState

UNET_ATTENTION_CONTEXT_WRAPPER_KEY = "simple_syrup.unet_regional_attention_contexts"


class StandardUnetAttentionContextDiffusionWrapper:
    """Adapt the installed UNet call contract to shared active context state."""

    def __init__(
        self,
        diffusion_model: object,
        state: StandardUnetAttentionState,
        *,
        model_call_resolver: RegionalAttentionModelCallResolver = (
            REGIONAL_ATTENTION_MODEL_CALL_RESOLVER
        ),
        operation_session: StandardUnetRegionalOperationSession | None = None,
    ) -> None:
        """Retain weak model identity and the shared plan/call authorities."""

        try:
            self._model: ReferenceType[object] = ref(diffusion_model)
        except TypeError as error:
            raise TypeError(
                "Standard UNet diffusion model must be weak-referenceable."
            ) from error
        if not isinstance(state, StandardUnetAttentionState):
            raise TypeError("Standard UNet context wrapper requires attention state.")
        if not isinstance(model_call_resolver, RegionalAttentionModelCallResolver):
            raise TypeError(
                "Standard UNet context wrapper requires a model-call resolver."
            )
        self._state = state
        self._model_call_resolver = model_call_resolver
        if operation_session is not None and not isinstance(
            operation_session,
            StandardUnetRegionalOperationSession,
        ):
            raise TypeError("Standard UNet operation session has an invalid type.")
        self._operation_session = operation_session

    def __call__(
        self,
        executor: DiffusionWrapperExecutor,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Resolve and publish contexts for exactly one nested UNet execution."""

        if self._model() is not executor.class_obj:
            raise ValueError(
                "Standard UNet context wrapper executor does not own the bound model."
            )
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
        contexts = self._model_call_resolver.resolve(
            self._state.plan,
            model_input=args[0],
            base_context=args[2],
            transformer_options=args[5],
        )
        forwarded_args = (*args[:2], contexts.base_context, *args[3:])
        with (
            self._state.execution_context.activate(contexts),
            self._state.resolution_cache.activate(),
        ):
            if self._operation_session is not None:
                with (
                    self._operation_session.activate(contexts, args[5]),
                    REGIONAL_OPERATION_INVOCATION_CONTEXT.activate(
                        self._operation_session
                    ),
                ):
                    return executor(*forwarded_args, **kwargs)
            return executor(*forwarded_args, **kwargs)


def unet_attention_context_wrapper_mutation(
    diffusion_model: object,
    state: StandardUnetAttentionState,
    *,
    operation_session: StandardUnetRegionalOperationSession | None = None,
) -> ModelDiffusionWrapperMutation:
    """Return the clone-local standard-UNet context wrapper mutation."""

    return ModelDiffusionWrapperMutation(
        UNET_ATTENTION_CONTEXT_WRAPPER_KEY,
        StandardUnetAttentionContextDiffusionWrapper(
            diffusion_model,
            state,
            operation_session=operation_session,
        ),
    )
