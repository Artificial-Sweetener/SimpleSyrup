# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble standard-UNet regional plans and loaded-call operation scope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import comfy.model_patcher
from comfy.patcher_extension import CallbacksMP
from torch import nn

from .convolution_execution_plan import (
    REGIONAL_CONVOLUTION_EXECUTION_PLAN_FACTORY,
    RegionalConvolutionExecutionPlanFactory,
)
from .execution_cache import RegionalLoraExecutionCache
from .host_operation_backing import (
    RegionalConvolutionOperationPatch,
    RegionalLinearOperationPatch,
)
from .linear_execution_plan import (
    REGIONAL_LINEAR_EXECUTION_PLAN_FACTORY,
    RegionalLinearExecutionPlanFactory,
)
from .operation_cache_lifecycle import (
    REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
    RegionalOperationCacheLifecycle,
    RegionalOperationPlan,
)
from .operation_call_scope import (
    RegionalCallScopedOperation,
    RegionalOperationCallScope,
)
from .operation_invocation import (
    REGIONAL_OPERATION_INVOCATION_CONTEXT,
    RegionalOperationInvocationContext,
)
from .target_binding import (
    BoundRegionalLoraModuleClass,
    BoundRegionalLoraOperation,
    RegionalLoraBindingResult,
)


@dataclass(frozen=True, slots=True)
class RegionalOperationAssembly:
    """Retain loaded-call wrappers and their shared detach lifecycle."""

    call_scope: RegionalOperationCallScope
    cache_lifecycle: RegionalOperationCacheLifecycle

    def __post_init__(self) -> None:
        """Require the two focused runtime owners."""

        if not isinstance(self.call_scope, RegionalOperationCallScope):
            raise TypeError("Regional assembly call scope is invalid.")
        if not isinstance(self.cache_lifecycle, RegionalOperationCacheLifecycle):
            raise TypeError("Regional assembly cache lifecycle is invalid.")


class RegionalOperationAssembler:
    """Own target grouping, static plans, and call-scoped wrapper construction."""

    def __init__(
        self,
        *,
        linear_factory: RegionalLinearExecutionPlanFactory = (
            REGIONAL_LINEAR_EXECUTION_PLAN_FACTORY
        ),
        convolution_factory: RegionalConvolutionExecutionPlanFactory = (
            REGIONAL_CONVOLUTION_EXECUTION_PLAN_FACTORY
        ),
        context: RegionalOperationInvocationContext = (
            REGIONAL_OPERATION_INVOCATION_CONTEXT
        ),
    ) -> None:
        """Retain focused plan and task-local invocation collaborators."""

        self._linear_factory = linear_factory
        self._convolution_factory = convolution_factory
        self._context = context

    def assemble(
        self,
        binding: RegionalLoraBindingResult,
        *,
        model: comfy.model_patcher.ModelPatcher,
        cache: RegionalLoraExecutionCache,
    ) -> RegionalOperationAssembly:
        """Return plans and wrappers without changing pending object patches."""

        self._validate_inputs(binding, model=model, cache=cache)
        graph_root = model.model
        if not isinstance(graph_root, nn.Module):
            raise TypeError("Regional operation assembly requires a module graph.")
        operations: list[RegionalCallScopedOperation] = []
        plans: list[RegionalOperationPlan] = []
        for module_path, entries in _group_by_module(binding.entries).items():
            module = entries[0].module
            if not isinstance(module, nn.Module):
                raise AssertionError("Admissible binding lost its target module.")
            module_class = entries[0].module_class
            if module_class is BoundRegionalLoraModuleClass.LINEAR:
                linear_plan = self._linear_factory.build(
                    entries,
                    model=model,
                    cache=cache,
                )
                plan: RegionalOperationPlan = linear_plan
                replacement: nn.Module = RegionalLinearOperationPatch(
                    module_path,
                    module,
                    linear_plan,
                    context=self._context,
                )
            elif module_class in (
                BoundRegionalLoraModuleClass.CONVOLUTION_1D,
                BoundRegionalLoraModuleClass.CONVOLUTION_2D,
                BoundRegionalLoraModuleClass.CONVOLUTION_3D,
            ):
                convolution_plan = self._convolution_factory.build(entries)
                plan = convolution_plan
                replacement = RegionalConvolutionOperationPatch(
                    module_path,
                    module,
                    convolution_plan,
                    context=self._context,
                )
            else:
                raise ValueError(
                    f"Regional operation module {module_path!r} is unsupported."
                )
            plans.append(plan)
            operations.append(
                RegionalCallScopedOperation(module_path, module, replacement)
            )
        return RegionalOperationAssembly(
            RegionalOperationCallScope(graph_root, tuple(operations)),
            RegionalOperationCacheLifecycle(tuple(plans), cache),
        )

    @staticmethod
    def _validate_inputs(
        binding: object,
        *,
        model: object,
        cache: object,
    ) -> None:
        """Reject incomplete, injected, or collision-prone model state."""

        if not isinstance(binding, RegionalLoraBindingResult):
            raise TypeError("Regional operation assembly requires binding evidence.")
        if not binding.admissible or not binding.entries:
            raise ValueError("Regional operation assembly requires complete admission.")
        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError("Regional operation assembly requires a ModelPatcher.")
        if not isinstance(cache, RegionalLoraExecutionCache):
            raise TypeError("Regional operation assembly requires an execution cache.")
        if model.is_injected:
            raise ValueError("Regional operation assembly requires an ejected MODEL.")
        injections = model.injections
        if not isinstance(injections, Mapping):
            raise TypeError("MODEL injections must be a mapping.")
        if any(values for values in injections.values()):
            raise ValueError(
                "Regional operations cannot compose with existing MODEL injections."
            )
        callbacks = model.get_callbacks(
            CallbacksMP.ON_DETACH,
            REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
        )
        if not isinstance(callbacks, list):
            raise TypeError("MODEL detach callback state must be a list.")
        if callbacks:
            raise ValueError("Regional operation detach callback is already installed.")
        active_paths = tuple(
            path
            for entry in binding.entries
            for path in entry.active_object_patch_paths
        )
        if active_paths:
            raise ValueError(
                "Regional operation targets overlap existing MODEL object patches: "
                f"{tuple(dict.fromkeys(active_paths))!r}."
            )


def _group_by_module(
    entries: tuple[BoundRegionalLoraOperation, ...],
) -> Mapping[str, tuple[BoundRegionalLoraOperation, ...]]:
    """Group declared-order adapter uses by one exact native module."""

    grouped: dict[str, list[BoundRegionalLoraOperation]] = {}
    identities: dict[str, tuple[object, BoundRegionalLoraModuleClass]] = {}
    for entry in entries:
        path = entry.descriptor.target.model_target
        identity = (entry.module, entry.module_class)
        existing = identities.get(path)
        if existing is not None and (
            existing[0] is not identity[0] or existing[1] is not identity[1]
        ):
            raise ValueError(
                f"Regional operation path {path!r} has inconsistent ownership."
            )
        identities[path] = identity
        grouped.setdefault(path, []).append(entry)
    return {path: tuple(values) for path, values in grouped.items()}


REGIONAL_OPERATION_ASSEMBLER = RegionalOperationAssembler()
