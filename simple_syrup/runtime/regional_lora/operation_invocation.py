# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish task-local runtime values for installed regional operations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ...masking.regional_activation_mask_projection import RegionalActivationMaskBatch


@dataclass(frozen=True, slots=True)
class RegionalOperationInvocation:
    """Retain masks and ordered schedule strengths for one target module call."""

    masks: RegionalActivationMaskBatch
    schedule_strengths: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require typed masks and an explicit immutable strength sequence."""

        if not isinstance(self.masks, RegionalActivationMaskBatch):
            raise TypeError("Regional operation invocation requires activation masks.")
        if not isinstance(self.schedule_strengths, tuple):
            raise TypeError("Regional operation schedule strengths must be a tuple.")


class RegionalOperationInvocationContext:
    """Own target-addressed regional operation values for one nested model call."""

    def __init__(self) -> None:
        """Create one empty task-local invocation map."""

        self._current: ContextVar[Mapping[str, RegionalOperationInvocation] | None] = (
            ContextVar("simple_syrup_regional_operation_invocations", default=None)
        )

    def current_for(self, module_path: str) -> RegionalOperationInvocation | None:
        """Return the active target invocation or no value outside its scope."""

        current = self._current.get()
        return None if current is None else current.get(module_path)

    @contextmanager
    def activate(
        self,
        invocations: Mapping[str, RegionalOperationInvocation],
    ) -> Iterator[None]:
        """Publish one immutable-by-contract target map for a nested execution."""

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
        token = self._current.set(copied)
        try:
            yield
        finally:
            self._current.reset(token)


REGIONAL_OPERATION_INVOCATION_CONTEXT = RegionalOperationInvocationContext()
