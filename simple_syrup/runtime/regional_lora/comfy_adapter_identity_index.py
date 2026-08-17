# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Index exact source-object identities retained by normalized Comfy adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from comfy.weight_adapter.base import WeightAdapterBase


@dataclass(frozen=True, slots=True)
class ComfyAdapterIdentityIndex:
    """Retain reachable object identities without equality or strong references."""

    _identities: frozenset[int]

    @classmethod
    def build(cls, operations: Iterable[object]) -> ComfyAdapterIdentityIndex:
        """Walk each admitted host container edge once and tolerate cycles."""

        identities: set[int] = set()
        expanded: set[int] = set()
        for operation in operations:
            _collect_identities(operation, identities, expanded)
        return cls(frozenset(identities))

    def contains(self, value: object) -> bool:
        """Report whether the exact live object occurs in normalized evidence."""

        return id(value) in self._identities


def _collect_identities(
    value: object,
    identities: set[int],
    expanded: set[int],
) -> None:
    """Traverse only containers admitted by the established evidence contract."""

    identity = id(value)
    identities.add(identity)
    if identity in expanded:
        return
    if isinstance(value, WeightAdapterBase):
        expanded.add(identity)
        _collect_identities(value.weights, identities, expanded)
        return
    if isinstance(value, tuple | list):
        expanded.add(identity)
        for item in value:
            _collect_identities(item, identities, expanded)
