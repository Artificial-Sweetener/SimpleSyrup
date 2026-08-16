# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own task-local execution for one stable standard-UNet variant root."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .standard_unet_variant_execution import StandardUnetVariantExecution


class StandardUnetVariantExecutionSession:
    """Bind request-local composition to a cache-stable diffusion root."""

    def __init__(self) -> None:
        """Create one task-local execution slot."""

        self._current: ContextVar[StandardUnetVariantExecution | None] = ContextVar(
            f"simple_syrup_standard_unet_variant_execution_{id(self)}",
            default=None,
        )

    @contextmanager
    def activate(
        self,
        execution: StandardUnetVariantExecution,
    ) -> Iterator[None]:
        """Activate one request execution and restore prior nested state."""

        if not isinstance(execution, StandardUnetVariantExecution):
            raise TypeError("Standard UNet variant session requires execution.")
        token = self._current.set(execution)
        try:
            yield
        finally:
            self._current.reset(token)

    def require_current(self) -> StandardUnetVariantExecution:
        """Return the active request execution or fail before graph work."""

        execution = self._current.get()
        if execution is None:
            raise RuntimeError("Standard UNet variant execution is not active.")
        return execution


class StandardUnetVariantExecutionActivation:
    """Activate one request execution through Comfy's diffusion wrapper stack."""

    def __init__(
        self,
        session: StandardUnetVariantExecutionSession,
        execution: StandardUnetVariantExecution,
    ) -> None:
        """Retain the stable session and request-local execution."""

        if not isinstance(session, StandardUnetVariantExecutionSession):
            raise TypeError("Standard UNet variant activation requires a session.")
        if not isinstance(execution, StandardUnetVariantExecution):
            raise TypeError("Standard UNet variant activation requires execution.")
        self._session = session
        self._execution = execution

    def __call__(
        self,
        executor: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Run the remaining wrapper stack inside request-local activation."""

        with self._session.activate(self._execution):
            return executor(*args, **kwargs)
