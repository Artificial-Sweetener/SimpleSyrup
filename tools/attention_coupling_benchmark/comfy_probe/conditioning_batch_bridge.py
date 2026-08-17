# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize canonical ConditioningBatch values across Comfy module namespaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from simple_syrup.domain.conditioning_batch import ConditioningBatch


@runtime_checkable
class _ConditioningBatchValue(Protocol):
    """Describe the immutable domain surface across Comfy loader namespaces."""

    @property
    def entries(self) -> tuple[object, ...]:
        """Return ordered conditioning values."""

        ...


def normalize_conditioning_batch(value: object) -> object:
    """Return one local batch for canonical host values or pass conditioning through."""

    if isinstance(value, ConditioningBatch):
        return value
    if not hasattr(value, "entries"):
        return value
    value_type = type(value)
    if value_type.__name__ != "ConditioningBatch" or not value_type.__module__.endswith(
        ".domain.conditioning_batch"
    ):
        raise TypeError("Unsupported conditioning batch runtime type.")
    if not isinstance(value, _ConditioningBatchValue):
        raise TypeError("ConditioningBatch must expose immutable entries.")
    entries = value.entries
    if not isinstance(entries, tuple):
        raise TypeError("ConditioningBatch entries must be an immutable tuple.")
    if not entries:
        raise ValueError("ConditioningBatch entries must not be empty.")
    return ConditioningBatch(entries)
