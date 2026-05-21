# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file integrate AUTOMATIC1111 sampler behavior. See
# third_party/manifest.toml and third_party/NOTICE.md.

"""Sampler resolution helpers for SimpleSyrup sampling nodes."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

from ..shared.logging import get_logger
from .a1111_sampling import sample_euler_ancestral_a1111

LOGGER = get_logger(__name__)
EXTRA_SAMPLERS = ("euler_a_a1111",)


class SamplerObject(Protocol):
    """Represent the executable sampler object returned by ComfyUI."""

    def sample(self, *args: object, **kwargs: object) -> object:
        """Run the ComfyUI sampler."""


def available_samplers() -> tuple[str, ...]:
    """Return core ComfyUI samplers plus locally resolved extra samplers."""

    comfy_samplers = _comfy_samplers()
    core_samplers = tuple(str(name) for name in comfy_samplers.KSampler.SAMPLERS)
    return _unique_sampler_names(core_samplers + EXTRA_SAMPLERS)


def resolve_sampler(sampler_name: str) -> SamplerObject:
    """Return a ComfyUI sampler object for a validated sampler name."""

    supported_samplers = available_samplers()
    if sampler_name not in supported_samplers:
        supported = ", ".join(supported_samplers)
        LOGGER.error(
            "Unsupported sampler requested",
            extra={
                "operation": "resolve_sampler",
                "sampler_name": sampler_name,
                "supported_samplers": supported,
            },
        )
        raise ValueError(
            f"Unsupported sampler '{sampler_name}'. Supported samplers are: {supported}"
        )

    if sampler_name in EXTRA_SAMPLERS:
        return _resolve_extra_sampler(sampler_name)

    return cast(SamplerObject, _comfy_samplers().sampler_object(sampler_name))


def _resolve_extra_sampler(sampler_name: str) -> SamplerObject:
    """Resolve a SimpleSyrup-owned sampler object."""

    if sampler_name == "euler_a_a1111":
        return cast(
            SamplerObject,
            _comfy_samplers().KSAMPLER(sample_euler_ancestral_a1111),
        )
    raise ValueError(f"Unsupported extra sampler '{sampler_name}'.")


def _unique_sampler_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return sampler names in first-seen order without duplicates."""

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        unique_names.append(name)
        seen.add(name)
    return tuple(unique_names)


def _comfy_samplers() -> ModuleType:
    """Import ComfyUI samplers lazily to keep registration imports lightweight."""

    return import_module("comfy.samplers")
