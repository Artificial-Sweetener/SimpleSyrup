# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute regional deltas around already-prepared native Comfy operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from .convolution_execution import (
    REGIONAL_CONVOLUTION_EXECUTOR,
    RegionalConvolutionExecutor,
)
from .convolution_execution_plan import RegionalConvolutionExecutionPlan
from .host_linear_parameters import RegionalHostLinearParameterProvider
from .linear_execution import REGIONAL_LINEAR_EXECUTOR, RegionalLinearExecutor
from .linear_execution_plan import RegionalLinearExecutionPlan
from .operation_invocation import (
    REGIONAL_OPERATION_INVOCATION_CONTEXT,
    RegionalOperationInvocationContext,
)


class RegionalLinearOperationPatch(nn.Module):
    """Execute one live native Linear plus task-local regional deltas."""

    def __init__(
        self,
        module_path: str,
        original: nn.Module,
        plan: RegionalLinearExecutionPlan,
        *,
        context: RegionalOperationInvocationContext = (
            REGIONAL_OPERATION_INVOCATION_CONTEXT
        ),
        executor: RegionalLinearExecutor = REGIONAL_LINEAR_EXECUTOR,
    ) -> None:
        """Bind a static plan without registering or copying the native module."""

        super().__init__()
        _validate_patch_inputs(module_path, original, plan, context)
        if not isinstance(executor, RegionalLinearExecutor):
            raise TypeError("Regional Linear operation requires a typed executor.")
        self.__dict__["_original"] = original
        self.__dict__["_native_forward"] = original.forward
        self._linear_parameters = RegionalHostLinearParameterProvider(original)
        self._module_path = module_path
        self._plan = plan
        self._context = context
        self._executor = executor

    @property
    def original(self) -> nn.Module:
        """Return the unregistered native operation used during call scope."""

        original = self.__dict__.get("_original")
        if not isinstance(original, nn.Module):
            raise RuntimeError("Regional Linear native operation is unavailable.")
        return original

    @property
    def native_forward(self) -> Callable[..., object]:
        """Return the exact native callable captured before call-scope override."""

        native_forward = self.__dict__.get("_native_forward")
        if not callable(native_forward):
            raise RuntimeError("Regional Linear native forward is unavailable.")
        return cast(Callable[..., object], native_forward)

    def forward(self, inputs: torch.Tensor, *args: object, **kwargs: object) -> object:
        """Delegate unchanged when inactive or execute the explicit regional call."""

        invocation = self._context.resolve(self._module_path, self._plan, inputs)
        if invocation is None:
            return self.native_forward(inputs, *args, **kwargs)
        return self._executor.execute(
            self.native_forward,
            inputs,
            *args,
            plan=self._plan,
            masks=invocation.masks,
            schedule_strengths=invocation.schedule_strengths,
            parameter_provider=self._linear_parameters,
            **kwargs,
        )


class RegionalConvolutionOperationPatch(nn.Module):
    """Execute one live native convolution plus task-local regional deltas."""

    def __init__(
        self,
        module_path: str,
        original: nn.Module,
        plan: RegionalConvolutionExecutionPlan,
        *,
        context: RegionalOperationInvocationContext = (
            REGIONAL_OPERATION_INVOCATION_CONTEXT
        ),
        executor: RegionalConvolutionExecutor = REGIONAL_CONVOLUTION_EXECUTOR,
    ) -> None:
        """Bind a static plan without registering or copying the native module."""

        super().__init__()
        _validate_patch_inputs(module_path, original, plan, context)
        if not isinstance(executor, RegionalConvolutionExecutor):
            raise TypeError("Regional convolution operation requires a typed executor.")
        self.__dict__["_original"] = original
        self.__dict__["_native_forward"] = original.forward
        self._module_path = module_path
        self._plan = plan
        self._context = context
        self._executor = executor

    @property
    def original(self) -> nn.Module:
        """Return the unregistered native operation used during call scope."""

        original = self.__dict__.get("_original")
        if not isinstance(original, nn.Module):
            raise RuntimeError("Regional convolution native operation is unavailable.")
        return original

    @property
    def native_forward(self) -> Callable[..., object]:
        """Return the exact native callable captured before call-scope override."""

        native_forward = self.__dict__.get("_native_forward")
        if not callable(native_forward):
            raise RuntimeError("Regional convolution native forward is unavailable.")
        return cast(Callable[..., object], native_forward)

    def forward(self, inputs: torch.Tensor, *args: object, **kwargs: object) -> object:
        """Delegate unchanged when inactive or execute the regional convolution."""

        invocation = self._context.resolve(self._module_path, self._plan, inputs)
        if invocation is None:
            return self.native_forward(inputs, *args, **kwargs)
        return self._executor.execute(
            self.native_forward,
            inputs,
            *args,
            plan=self._plan,
            masks=invocation.masks,
            schedule_strengths=invocation.schedule_strengths,
            **kwargs,
        )


def _validate_patch_inputs(
    module_path: object,
    original: object,
    plan: object,
    context: object,
) -> None:
    """Validate one native operation and static plan before retaining them."""

    if not isinstance(module_path, str) or not module_path.strip():
        raise ValueError("Regional operation module path must be nonempty.")
    if not isinstance(original, nn.Module):
        raise TypeError("Regional operation original must be a PyTorch module.")
    if not isinstance(
        plan,
        RegionalLinearExecutionPlan | RegionalConvolutionExecutionPlan,
    ):
        raise TypeError("Regional operation requires a typed execution plan.")
    if not isinstance(context, RegionalOperationInvocationContext):
        raise TypeError("Regional operation requires an invocation context.")
