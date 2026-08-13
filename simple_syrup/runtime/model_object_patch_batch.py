# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install one preflighted batch of exact clone-local model object patches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import cast


@dataclass(frozen=True, slots=True)
class ExactModelObjectReplacement:
    """Describe one exact expected object and its clone-local replacement."""

    path: str
    expected_object: object
    replacement: object

    def __post_init__(self) -> None:
        """Require one canonical nonempty dotted model path."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or any(not segment for segment in self.path.split("."))
        ):
            raise ValueError("MODEL object replacement requires a dotted path.")


@dataclass(frozen=True, slots=True)
class ModelObjectPatchBatchMutation:
    """Preflight every exact replacement before changing pending patch state."""

    replacements: tuple[ExactModelObjectReplacement, ...]

    def __post_init__(self) -> None:
        """Reject empty, invalid, duplicate, or hierarchically overlapping paths."""

        if not isinstance(self.replacements, tuple) or not self.replacements:
            raise ValueError("MODEL object patch batch cannot be empty.")
        if any(
            not isinstance(replacement, ExactModelObjectReplacement)
            for replacement in self.replacements
        ):
            raise TypeError("MODEL object patch batch contains an invalid value.")
        paths = tuple(replacement.path for replacement in self.replacements)
        if len(paths) != len(set(paths)):
            raise ValueError("MODEL object patch batch paths must be unique.")
        overlaps = _overlapping_paths(paths)
        if overlaps:
            raise ValueError(
                f"MODEL object patch batch paths cannot overlap: {overlaps!r}."
            )

    def apply(self, model: object) -> None:
        """Validate the complete clone surface, then install the entire batch."""

        getter = _required_method(model, "get_model_object", ("name",))
        adder = _required_method(model, "add_object_patch", ("name", "obj"))
        object_patches = _required_dictionary(model, "object_patches")
        backups = _required_dictionary(model, "object_patches_backup")
        paths = tuple(replacement.path for replacement in self.replacements)
        exact_collisions = tuple(
            path for path in paths if path in object_patches or path in backups
        )
        if exact_collisions:
            raise ValueError(
                "MODEL object patch batch collides with existing patch state: "
                f"{exact_collisions!r}."
            )
        existing_paths = tuple(
            path for path in (*object_patches, *backups) if isinstance(path, str)
        )
        overlaps = _overlapping_paths((*paths, *existing_paths), cross_only=paths)
        if overlaps:
            raise ValueError(
                "MODEL object patch batch collides with existing patch state: "
                f"{overlaps!r}."
            )
        for replacement in self.replacements:
            current = getter(replacement.path)
            if current is not replacement.expected_object:
                raise ValueError(
                    f"MODEL object path '{replacement.path}' does not match its "
                    "expected object."
                )
        original_patches = object_patches.copy()
        try:
            for replacement in self.replacements:
                adder(replacement.path, replacement.replacement)
        except BaseException:
            object_patches.clear()
            object_patches.update(original_patches)
            raise


def _required_method(
    model: object,
    name: str,
    parameters: tuple[str, ...],
) -> Callable[..., object]:
    """Return one exact installed positional ModelPatcher method."""

    method = getattr(model, name, None)
    if not callable(method):
        raise TypeError(f"MODEL must expose callable {name}({', '.join(parameters)}).")
    try:
        observed = tuple(signature(method).parameters.values())
    except (TypeError, ValueError) as error:
        raise TypeError(f"MODEL {name} signature cannot be inspected.") from error
    if len(observed) != len(parameters) or any(
        parameter.name != expected
        or parameter.kind
        not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for parameter, expected in zip(observed, parameters, strict=True)
    ):
        raise TypeError(f"MODEL {name} has an unsupported signature.")
    return cast(Callable[..., object], method)


def _required_dictionary(model: object, name: str) -> dict[object, object]:
    """Return one exact mutable ModelPatcher state dictionary."""

    value = getattr(model, name, None)
    if not isinstance(value, dict):
        raise TypeError(f"MODEL {name} must be a dictionary.")
    return value


def _overlapping_paths(
    paths: tuple[str, ...],
    *,
    cross_only: tuple[str, ...] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return equal or ancestor-related path pairs in deterministic order."""

    owned = set(cross_only) if cross_only is not None else None
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if owned is not None and (left in owned) == (right in owned):
                continue
            if (
                left == right
                or left.startswith(f"{right}.")
                or right.startswith(f"{left}.")
            ):
                ordered = sorted((left, right))
                pair = (ordered[0], ordered[1])
                if pair not in pairs:
                    pairs.append(pair)
    return tuple(sorted(pairs))
