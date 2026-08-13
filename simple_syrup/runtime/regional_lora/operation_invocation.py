# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish task-local runtime values for installed regional operations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .convolution_execution_plan import RegionalConvolutionExecutionPlan
from .linear_execution_plan import RegionalLinearExecutionPlan
from .operation_mask_resolution import RegionalOperationMaskBatch


@dataclass(frozen=True, slots=True)
class RegionalOperationInvocation:
    """Retain masks and ordered schedule strengths for one target module call."""

    masks: RegionalOperationMaskBatch
    schedule_strengths: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require typed masks and an explicit immutable strength sequence."""

        if not isinstance(self.masks, RegionalOperationMaskBatch):
            raise TypeError("Regional operation invocation requires use masks.")
        if not isinstance(self.schedule_strengths, tuple):
            raise TypeError("Regional operation schedule strengths must be a tuple.")


RegionalOperationExecutionPlan = (
    RegionalLinearExecutionPlan | RegionalConvolutionExecutionPlan
)


@runtime_checkable
class RegionalOperationInvocationResolver(Protocol):
    """Resolve one installed operation from exact live call evidence."""

    def resolve(
        self,
        module_path: str,
        plan: RegionalOperationExecutionPlan,
        inputs: torch.Tensor,
    ) -> RegionalOperationInvocation | None:
        """Return one active invocation or no regional work for this call."""

        ...


class StaticRegionalOperationInvocationResolver:
    """Serve immutable target values for deterministic executor characterization."""

    def __init__(
        self,
        invocations: Mapping[str, RegionalOperationInvocation],
    ) -> None:
        """Validate and copy one target-addressed characterization map."""

        if not isinstance(invocations, Mapping):
            raise TypeError("Regional operation invocations must be a mapping.")
        copied = dict(invocations)
        if any(not isinstance(path, str) or not path for path in copied):
            raise ValueError("Regional operation invocation paths must be nonempty.")
        if any(
            not isinstance(value, RegionalOperationInvocation)
            for value in copied.values()
        ):
            raise TypeError("Regional operation invocation values are invalid.")
        self._invocations = copied

    def resolve(
        self,
        module_path: str,
        plan: RegionalOperationExecutionPlan,
        inputs: torch.Tensor,
    ) -> RegionalOperationInvocation | None:
        """Return the characterized value after validating live boundary types."""

        if not isinstance(module_path, str) or not module_path:
            raise ValueError("Regional operation module path must be nonempty.")
        if not isinstance(
            plan,
            RegionalLinearExecutionPlan | RegionalConvolutionExecutionPlan,
        ):
            raise TypeError("Regional operation plan has an invalid type.")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Regional operation inputs must be a tensor.")
        return self._invocations.get(module_path)


class RegionalOperationInvocationContext:
    """Own one invocation resolver for a nested model call."""

    def __init__(self) -> None:
        """Create one empty task-local resolver slot."""

        self._current: ContextVar[RegionalOperationInvocationResolver | None] = (
            ContextVar("simple_syrup_regional_operation_resolver", default=None)
        )

    def resolve(
        self,
        module_path: str,
        plan: RegionalOperationExecutionPlan,
        inputs: torch.Tensor,
    ) -> RegionalOperationInvocation | None:
        """Delegate exact call evidence or return no value outside active scope."""

        current = self._current.get()
        return None if current is None else current.resolve(module_path, plan, inputs)

    @contextmanager
    def activate(
        self,
        resolver: RegionalOperationInvocationResolver,
    ) -> Iterator[None]:
        """Publish one typed invocation owner for a nested execution."""

        if not isinstance(resolver, RegionalOperationInvocationResolver):
            raise TypeError("Regional operation resolver has an invalid type.")
        token = self._current.set(resolver)
        try:
            yield
        finally:
            self._current.reset(token)


REGIONAL_OPERATION_INVOCATION_CONTEXT = RegionalOperationInvocationContext()
