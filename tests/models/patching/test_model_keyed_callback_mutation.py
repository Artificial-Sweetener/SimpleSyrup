# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove collision-safe Comfy MODEL callback registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from comfy.patcher_extension import CallbacksMP

from simple_syrup.runtime.model_patcher_mutations import ModelKeyedCallbackMutation


def test_keyed_callback_mutation_installs_exact_callback_on_real_patcher() -> None:
    """Register through Comfy's public keyed surface without changing identity."""

    model = _patcher()

    def callback(patcher: object, unpatch_all: bool) -> None:
        """Accept the installed detach contract."""

        del patcher, unpatch_all

    ModelKeyedCallbackMutation(
        CallbacksMP.ON_DETACH,
        "simple_syrup.device_cache",
        callback,
    ).apply(model)

    assert model.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.device_cache",
    ) == [callback]


def test_keyed_callback_mutation_rejects_collision_before_mutating() -> None:
    """Keep one authoritative owner for each namespaced callback key."""

    model = _patcher()

    def first(*args: object) -> tuple[object, ...]:
        """Return the first callback arguments."""

        return args

    def second(*args: object) -> tuple[object, ...]:
        """Return the second callback arguments."""

        return args

    mutation = ModelKeyedCallbackMutation(
        CallbacksMP.ON_DETACH,
        "simple_syrup.device_cache",
        first,
    )
    mutation.apply(model)

    with pytest.raises(ValueError, match="already installed"):
        ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            "simple_syrup.device_cache",
            second,
        ).apply(model)

    assert model.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.device_cache",
    ) == [first]


@pytest.mark.parametrize(
    ("callback_type", "key", "callback", "error_type", "message"),
    [
        ("", "simple_syrup.valid", lambda: None, ValueError, "non-empty"),
        ("event", "foreign.valid", lambda: None, ValueError, "must start"),
        ("event", "simple_syrup.", lambda: None, ValueError, "namespaced"),
        ("event", "simple_syrup.valid", None, TypeError, "must be callable"),
    ],
)
def test_keyed_callback_mutation_validates_owned_fields(
    callback_type: str,
    key: str,
    callback: Callable[..., object] | None,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject malformed declarations before inspecting Comfy callback state."""

    model = _patcher()

    with pytest.raises(error_type, match=message):
        ModelKeyedCallbackMutation(
            callback_type,
            key,
            cast(Callable[..., object], callback),
        ).apply(model)

    assert model.get_all_callbacks(callback_type) == []


def _patcher() -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(torch.nn.Linear(2, 2), device, device)
