# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fail-closed live ownership for keyed diffusion wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from comfy.patcher_extension import WrappersMP

from simple_syrup.runtime.diffusion_wrapper_invocation import (
    DIFFUSION_WRAPPER_INVOCATION_VALIDATOR,
)

_KEY = "simple_syrup.test_diffusion_wrapper"


@dataclass
class _Executor:
    """Expose only the bound-object surface used by the validator."""

    class_obj: object

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Reject execution because validation never invokes the executor."""

        raise AssertionError((args, kwargs))


def _wrapper(*_: object, **__: object) -> object:
    """Provide one stable callable wrapper identity."""

    return object()


def _options(registered: list[object]) -> dict[object, object]:
    """Return a transformer-options wrapper registry for one key."""

    return {
        "wrappers": {WrappersMP.DIFFUSION_MODEL: {_KEY: registered}},
    }


def test_invocation_accepts_exact_keyed_wrapper_on_delegate_model() -> None:
    """Use live keyed registration instead of unstable module identity."""

    DIFFUSION_WRAPPER_INVOCATION_VALIDATOR.require_owned(
        _Executor(object()),
        _options([_wrapper]),
        key=_KEY,
        wrapper=_wrapper,
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "no wrapper registry"),
        ({"wrappers": {}}, "no diffusion-model registry"),
        (
            {"wrappers": {WrappersMP.DIFFUSION_MODEL: {_KEY: "not-a-list"}}},
            "malformed",
        ),
        (_options([lambda: None]), "does not own"),
        (_options([_wrapper, _wrapper]), "does not own"),
    ],
)
def test_invocation_rejects_missing_foreign_duplicate_and_malformed_state(
    options: dict[object, object],
    message: str,
) -> None:
    """Reject every registry state except one exact namespaced owner."""

    with pytest.raises(ValueError, match=message):
        DIFFUSION_WRAPPER_INVOCATION_VALIDATOR.require_owned(
            _Executor(object()),
            options,
            key=_KEY,
            wrapper=_wrapper,
        )


def test_invocation_rejects_unbound_executor_and_foreign_key() -> None:
    """Require both a bound diffusion call and the exact namespaced key."""

    with pytest.raises(ValueError, match="bound model"):
        DIFFUSION_WRAPPER_INVOCATION_VALIDATOR.require_owned(
            _Executor(None),
            _options([_wrapper]),
            key=_KEY,
            wrapper=_wrapper,
        )
    with pytest.raises(ValueError, match="namespaced"):
        DIFFUSION_WRAPPER_INVOCATION_VALIDATOR.require_owned(
            _Executor(object()),
            _options([_wrapper]),
            key="foreign.key",
            wrapper=_wrapper,
        )
