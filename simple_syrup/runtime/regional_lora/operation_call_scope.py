# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Override native operation forwards only inside a loaded model call."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock

from torch import nn


@dataclass(frozen=True, slots=True)
class RegionalCallScopedOperation:
    """Describe one native operation and its lightweight forward owner."""

    path: str
    original: nn.Module
    replacement: nn.Module
    native_forward: Callable[..., object] = field(init=False, repr=False)
    had_instance_forward: bool = field(init=False, repr=False)
    instance_forward: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Require a canonical path and two distinct typed operations."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or any(not segment for segment in self.path.split("."))
        ):
            raise ValueError("Call-scoped operation requires a dotted path.")
        if not isinstance(self.original, nn.Module) or not isinstance(
            self.replacement,
            nn.Module,
        ):
            raise TypeError("Call-scoped operations must be PyTorch modules.")
        if self.original is self.replacement:
            raise ValueError("Call-scoped replacement must differ from its original.")
        native_forward = self.original.forward
        if not callable(native_forward) or not callable(self.replacement.forward):
            raise TypeError("Call-scoped operations require callable forwards.")
        object.__setattr__(self, "native_forward", native_forward)
        object.__setattr__(
            self,
            "had_instance_forward",
            "forward" in self.original.__dict__,
        )
        object.__setattr__(
            self,
            "instance_forward",
            self.original.__dict__.get("forward"),
        )


class RegionalOperationCallScope:
    """Own atomic forward override and exact restoration around one call."""

    def __init__(
        self,
        root: nn.Module,
        operations: tuple[RegionalCallScopedOperation, ...],
    ) -> None:
        """Retain one native graph and a non-overlapping replacement batch."""

        if not isinstance(root, nn.Module):
            raise TypeError("Regional call scope requires a module root.")
        if not isinstance(operations, tuple) or not operations:
            raise ValueError("Regional call scope requires operations.")
        if any(
            not isinstance(value, RegionalCallScopedOperation) for value in operations
        ):
            raise TypeError("Regional call scope contains an invalid operation.")
        paths = tuple(operation.path for operation in operations)
        if len(paths) != len(set(paths)):
            raise ValueError("Regional call-scoped operation paths must be unique.")
        if any(
            left.startswith(f"{right}.") or right.startswith(f"{left}.")
            for index, left in enumerate(paths)
            for right in paths[index + 1 :]
        ):
            raise ValueError("Regional call-scoped operation paths cannot overlap.")
        self._root = root
        self._operations = operations
        self._lock = RLock()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Override forwards without changing the registered native graph."""

        with self._lock:
            for operation in self._operations:
                parent, name = self._resolve_parent(operation.path)
                if getattr(parent, name) is not operation.original:
                    raise RuntimeError(
                        f"Regional call-scoped path {operation.path!r} lost its "
                        "native operation identity."
                    )
                if not _same_callable(
                    operation.original.forward,
                    operation.native_forward,
                ):
                    raise RuntimeError(
                        f"Regional call-scoped path {operation.path!r} lost its "
                        "native forward identity."
                    )
            installed: list[RegionalCallScopedOperation] = []
            try:
                for operation in self._operations:
                    object.__setattr__(
                        operation.original,
                        "forward",
                        operation.replacement.forward,
                    )
                    installed.append(operation)
                yield
            finally:
                for operation in reversed(installed):
                    if operation.had_instance_forward:
                        object.__setattr__(
                            operation.original,
                            "forward",
                            operation.instance_forward,
                        )
                    else:
                        object.__delattr__(operation.original, "forward")

    @property
    def operation_count(self) -> int:
        """Return the immutable installed-operation cardinality."""

        return len(self._operations)

    @property
    def operations(self) -> tuple[RegionalCallScopedOperation, ...]:
        """Return immutable operation descriptions for diagnostics."""

        return self._operations

    def _resolve_parent(self, path: str) -> tuple[object, str]:
        """Resolve one exact parent without mutating the module graph."""

        segments = path.split(".")
        parent: object = self._root
        for segment in segments[:-1]:
            if not hasattr(parent, segment):
                raise ValueError(f"Regional call-scoped path {path!r} is missing.")
            parent = getattr(parent, segment)
        name = segments[-1]
        if not hasattr(parent, name):
            raise ValueError(f"Regional call-scoped path {path!r} is missing.")
        return parent, name


def _same_callable(left: object, right: object) -> bool:
    """Compare ordinary functions and freshly materialized bound methods."""

    left_owner = getattr(left, "__self__", None)
    right_owner = getattr(right, "__self__", None)
    left_function = getattr(left, "__func__", left)
    right_function = getattr(right, "__func__", right)
    return left_owner is right_owner and left_function is right_function
