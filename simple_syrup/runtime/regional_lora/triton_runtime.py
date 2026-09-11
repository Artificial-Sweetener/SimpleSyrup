# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve optional Triton backends without importing them on Torch paths."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import import_module
from threading import Lock

LOGGER = logging.getLogger(__name__)


class TritonRuntimeResolver:
    """Own thread-safe lazy backend imports and optional-package fallback."""

    def __init__(
        self,
        *,
        import_module: Callable[[str], object] = import_module,
    ) -> None:
        """Retain an injectable importer and empty process-lifetime cache."""

        if not callable(import_module):
            raise TypeError("Triton runtime importer must be callable.")
        self._import_module = import_module
        self._lock = Lock()
        self._backends: dict[str, object | None] = {}
        self._missing_warning_emitted = False

    def resolve(self, backend_module: str) -> object | None:
        """Return one cached backend or None only when Triton is absent."""

        if not isinstance(backend_module, str) or not backend_module:
            raise ValueError("Triton backend module must be a non-empty string.")
        with self._lock:
            if backend_module in self._backends:
                return self._backends[backend_module]
            try:
                backend = self._import_module(backend_module)
            except ModuleNotFoundError as error:
                if error.name != "triton" and not (
                    isinstance(error.name, str) and error.name.startswith("triton.")
                ):
                    raise RuntimeError(
                        f"Triton backend {backend_module!r} failed to initialize."
                    ) from error
                backend = None
                if not self._missing_warning_emitted:
                    LOGGER.warning(
                        "Triton acceleration is unavailable; using the Torch "
                        "execution path",
                        extra={
                            "backend_module": backend_module,
                            "missing_dependency": error.name,
                        },
                    )
                    self._missing_warning_emitted = True
            except Exception as error:
                raise RuntimeError(
                    f"Triton backend {backend_module!r} failed to initialize."
                ) from error
            self._backends[backend_module] = backend
            return backend


TRITON_RUNTIME_RESOLVER = TritonRuntimeResolver()
