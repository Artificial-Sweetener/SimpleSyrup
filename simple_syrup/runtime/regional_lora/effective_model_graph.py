# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve read-only effective ModelPatcher objects through active object patches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import comfy.model_patcher


@dataclass(frozen=True, slots=True)
class EffectiveModelObject:
    """Retain one resolved object and every active patch prefix affecting it."""

    value: object
    active_object_patch_paths: tuple[str, ...]


class EffectiveModelGraphError(ValueError):
    """Report an absent or ambiguous effective graph path."""


class EffectiveModelGraphResolver:
    """Read effective graph objects without applying, ejecting, or mutating patches."""

    def resolve(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
        path: str,
    ) -> EffectiveModelObject:
        """Resolve one path through the most specific unambiguous active patch."""

        if not isinstance(patcher, comfy.model_patcher.ModelPatcher):
            raise TypeError("Effective model graph resolution requires a ModelPatcher.")
        if (
            not isinstance(path, str)
            or not path
            or any(not segment for segment in path.split("."))
        ):
            raise ValueError(
                "Effective model graph path must be a nonempty dotted path."
            )
        object_patches = _mapping(patcher.object_patches, name="object_patches")
        applicable = tuple(
            patch_path
            for patch_path in object_patches
            if isinstance(patch_path, str)
            and (path == patch_path or path.startswith(f"{patch_path}."))
        )
        nested = tuple(
            patch_path
            for patch_path in applicable
            if any(
                patch_path != other and patch_path.startswith(f"{other}.")
                for other in applicable
            )
        )
        if nested:
            paths = ", ".join(sorted(applicable))
            raise EffectiveModelGraphError(
                f"Effective model graph path {path!r} has overlapping object "
                f"patch prefixes: {paths}."
            )
        if applicable:
            patch_path = max(applicable, key=len)
            value = object_patches[patch_path]
            suffix = path.removeprefix(patch_path).removeprefix(".")
            if suffix:
                value = _traverse(value, suffix, full_path=path)
            return EffectiveModelObject(value, tuple(sorted(applicable)))
        try:
            value = patcher.get_model_object(path)
        except AttributeError as error:
            raise EffectiveModelGraphError(
                f"Effective model graph path {path!r} does not exist."
            ) from error
        return EffectiveModelObject(value, ())


def _traverse(value: object, suffix: str, *, full_path: str) -> object:
    """Traverse one suffix from an unapplied object-patch replacement."""

    current = value
    for segment in suffix.split("."):
        try:
            current = getattr(current, segment)
        except AttributeError as error:
            raise EffectiveModelGraphError(
                f"Effective model graph path {full_path!r} does not exist in its "
                "active object-patch replacement."
            ) from error
    return current


def _mapping(value: object, *, name: str) -> Mapping[object, object]:
    """Require the installed ModelPatcher mapping contract."""

    if not isinstance(value, Mapping):
        raise TypeError(f"ModelPatcher {name} must be a mapping.")
    return value


EFFECTIVE_MODEL_GRAPH_RESOLVER = EffectiveModelGraphResolver()
