# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preserve Comfy host state around clone-local regional operation execution."""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from torch import nn

from .convolution_execution import (
    REGIONAL_CONVOLUTION_EXECUTOR,
    RegionalConvolutionExecutor,
)
from .convolution_execution_plan import RegionalConvolutionExecutionPlan
from .linear_execution import REGIONAL_LINEAR_EXECUTOR, RegionalLinearExecutor
from .linear_execution_plan import RegionalLinearExecutionPlan
from .operation_invocation import (
    REGIONAL_OPERATION_INVOCATION_CONTEXT,
    RegionalOperationInvocationContext,
)

_COMFY_TRANSIENT_ATTRIBUTES = frozenset(
    {
        "comfy_cast_weights",
        "comfy_force_cast_weights",
        "comfy_patched_weights",
        "prev_comfy_cast_weights",
        "seed_key",
        "weight_function",
        "bias_function",
        "_pin_state",
        "_prefetch",
        "_v",
        "_v_signature",
    }
)
_COMFY_PARAMETER_SUFFIXES = (
    "_comfy_model_dtype",
    "_function",
    "_lowvram_function",
)


class RegionalHostBackedOperation(nn.Module):
    """Mirror public host state while retaining one private exact-class shell."""

    def __getattr__(self, name: str) -> Any:
        """Resolve registered state locally and original operation state privately."""

        try:
            return super().__getattr__(name)
        except AttributeError as error:
            backing = self.__dict__.get("_host_backing")
            if isinstance(backing, RegionalHostOperationBacking) and hasattr(
                backing.module, name
            ):
                return getattr(backing.module, name)
            raise error

    def __setattr__(self, name: str, value: Any) -> None:
        """Mirror Comfy writes and public operation state onto the exact shell."""

        backing = self.__dict__.get("_host_backing")
        if isinstance(backing, RegionalHostOperationBacking) and (
            backing.owns_public_attribute(name)
        ):
            backing.replace_public_attribute(self, name, value)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete transient host state from both public views."""

        backing = self.__dict__.get("_host_backing")
        if isinstance(backing, RegionalHostOperationBacking) and (
            backing.owns_public_attribute(name)
        ):
            if hasattr(backing.module, name):
                delattr(backing.module, name)
            if name in self.__dict__:
                super().__delattr__(name)
            return
        super().__delattr__(name)


class RegionalHostOperationBacking:
    """Own one source-isolated exact-class shell and its public state mirror."""

    def __init__(self, module: nn.Module) -> None:
        """Copy registration maps without copying parameter storage."""

        if not isinstance(module, nn.Module):
            raise TypeError("Regional host backing requires a PyTorch module.")
        self._module = self._execution_shell(module)
        self._direct_state_names: frozenset[str] = frozenset()
        self._public_attribute_names = frozenset(
            name for name in module.__dict__ if name not in _MODULE_INTERNAL_STATE
        )

    @property
    def module(self) -> nn.Module:
        """Return the private original exact-class execution shell."""

        return self._module

    def install_direct_state(self, replacement: nn.Module) -> None:
        """Expose parameters and buffers at their unchanged public model paths."""

        if "weight" in self._module._parameters:
            self._module._parameters.setdefault("bias", None)
        parameter_names = tuple(self._module._parameters)
        buffer_names = tuple(self._module._buffers)
        overlap = frozenset(parameter_names).intersection(buffer_names)
        if overlap:
            raise ValueError(
                "Regional host operation exposes duplicate parameter/buffer names: "
                f"{sorted(overlap)!r}."
            )
        replacement._parameters = self._module._parameters
        replacement._buffers = self._module._buffers
        replacement._non_persistent_buffers_set = (
            self._module._non_persistent_buffers_set
        )
        self._module._parameters = replacement._parameters
        self._module._buffers = replacement._buffers
        self._module._non_persistent_buffers_set = (
            replacement._non_persistent_buffers_set
        )
        self._direct_state_names = frozenset((*parameter_names, *buffer_names))
        replacement.training = self._module.training

    def owns_public_attribute(self, name: str) -> bool:
        """Report operation or Comfy host state mirrored to the exact shell."""

        if (
            name in self._direct_state_names
            or name in self._public_attribute_names
            or name in _COMFY_TRANSIENT_ATTRIBUTES
        ):
            return True
        return any(
            name == f"{parameter_name}{suffix}"
            for parameter_name in ("weight", "bias")
            for suffix in _COMFY_PARAMETER_SUFFIXES
        )

    def replace_public_attribute(
        self,
        replacement: nn.Module,
        name: str,
        value: Any,
    ) -> None:
        """Keep Comfy's installed module and exact shell synchronized."""

        setattr(self._module, name, value)
        nn.Module.__setattr__(replacement, name, value)

    @staticmethod
    def _execution_shell(module: nn.Module) -> nn.Module:
        """Return an exact-class shell isolated from source registration mutation."""

        shell = copy(module)
        shell._parameters = {
            name: None if parameter is None else copy(parameter)
            for name, parameter in module._parameters.items()
        }
        shell._buffers = {
            name: None if buffer is None else copy(buffer)
            for name, buffer in module._buffers.items()
        }
        shell._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        shell._modules = module._modules.copy()
        return shell


class RegionalLinearOperationPatch(RegionalHostBackedOperation):
    """Execute one host Linear plus task-local regional low-rank deltas."""

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
        """Bind one static plan while retaining task-local runtime inputs."""

        super().__init__()
        _validate_patch_inputs(module_path, original, plan, context)
        if not isinstance(executor, RegionalLinearExecutor):
            raise TypeError("Regional Linear operation requires a typed executor.")
        self._host_backing = RegionalHostOperationBacking(original)
        self._host_backing.install_direct_state(self)
        self._module_path = module_path
        self._plan = plan
        self._context = context
        self._executor = executor

    def forward(self, inputs: torch.Tensor, *args: object, **kwargs: object) -> object:
        """Delegate unchanged when inactive or execute the explicit regional call."""

        invocation = self._context.resolve(self._module_path, self._plan, inputs)
        if invocation is None:
            return self._host_backing.module(inputs, *args, **kwargs)
        return self._executor.execute(
            self._host_backing.module,
            inputs,
            *args,
            plan=self._plan,
            masks=invocation.masks,
            schedule_strengths=invocation.schedule_strengths,
            **kwargs,
        )


class RegionalConvolutionOperationPatch(RegionalHostBackedOperation):
    """Execute one host convolution plus task-local regional LoRA deltas."""

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
        """Bind one static convolution plan and host-compatible shell."""

        super().__init__()
        _validate_patch_inputs(module_path, original, plan, context)
        if not isinstance(executor, RegionalConvolutionExecutor):
            raise TypeError("Regional convolution operation requires a typed executor.")
        self._host_backing = RegionalHostOperationBacking(original)
        self._host_backing.install_direct_state(self)
        self._module_path = module_path
        self._plan = plan
        self._context = context
        self._executor = executor

    def forward(self, inputs: torch.Tensor, *args: object, **kwargs: object) -> object:
        """Delegate unchanged when inactive or execute the regional convolution."""

        invocation = self._context.resolve(self._module_path, self._plan, inputs)
        if invocation is None:
            return self._host_backing.module(inputs, *args, **kwargs)
        return self._executor.execute(
            self._host_backing.module,
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
    """Validate one static operation shell before retaining host state."""

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


_MODULE_INTERNAL_STATE = frozenset(
    {
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_modules",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)
