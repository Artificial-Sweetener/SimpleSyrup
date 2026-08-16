# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preserve Comfy diffusion wrappers around persistent variant execution."""

from __future__ import annotations

from comfy.patcher_extension import WrapperExecutor, WrappersMP, get_all_wrappers
from torch import nn

from .standard_unet_variant_execution_session import (
    StandardUnetVariantExecutionSession,
)
from .standard_unet_variant_invocation import StandardUnetVariantInvocation


class StandardUnetVariantForward:
    """Adapt native UNet forward wrapping to one regional execution owner."""

    def __init__(
        self,
        root: nn.Module,
        execution_session: StandardUnetVariantExecutionSession,
    ) -> None:
        """Retain the exact wrapper class object and regional execution."""

        if not isinstance(root, nn.Module):
            raise TypeError("Standard UNet variant forward requires a root module.")
        if not isinstance(execution_session, StandardUnetVariantExecutionSession):
            raise TypeError("Standard UNet variant forward requires a session.")
        self._root = root
        self._execution_session = execution_session

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Run every installed diffusion wrapper around one composed prediction."""

        invocation = StandardUnetVariantInvocation.bind(args, kwargs)
        wrappers = get_all_wrappers(
            WrappersMP.DIFFUSION_MODEL,
            invocation.transformer_options,
        )
        return WrapperExecutor.new_class_executor(
            self._execute_current,
            self._root,
            wrappers,
        ).execute(*invocation.arguments, **invocation.keyword_arguments)

    def _execute_current(self, *args: object, **kwargs: object) -> object:
        """Delegate the stable root to the active request execution."""

        return self._execution_session.require_current().execute(*args, **kwargs)
