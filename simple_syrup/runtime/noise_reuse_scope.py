# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own one request-scoped noise handoff across existing sampler routing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ..domain.noise_reuse import NoiseReuseOptions, SamplingProvenance


@dataclass(slots=True)
class NoiseReuseExecution:
    """Keep a single receiver's immutable capture and consumption lifecycle."""

    options: NoiseReuseOptions
    provenance: SamplingProvenance
    consumed: bool = False


_CURRENT: ContextVar[NoiseReuseExecution | None] = ContextVar(
    "simple_syrup_noise_reuse", default=None
)


def current_noise_reuse() -> NoiseReuseExecution | None:
    """Resolve only the receiving service's current execution, never cached globals."""
    return _CURRENT.get()


def needs_finishing_model() -> bool:
    """Request a spatial inversion factory only for enabled finishing work."""
    current = _CURRENT.get()
    return current is not None and current.options.finishing_steps > 0


@contextmanager
def noise_reuse_scope(execution: NoiseReuseExecution | None) -> Iterator[None]:
    """Restore the caller's context even when validation or native inference fails."""
    token = _CURRENT.set(execution)
    try:
        yield
    finally:
        _CURRENT.reset(token)
