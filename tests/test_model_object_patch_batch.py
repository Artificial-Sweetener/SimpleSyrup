# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify atomic exact-object mutation batches against installed Comfy."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from simple_syrup.runtime.model_object_patch_batch import (
    ExactModelObjectReplacement,
    ModelObjectPatchBatchMutation,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE


def test_batch_preflights_every_identity_before_installing_any_path() -> None:
    """A late identity failure cannot leave one earlier target installed."""

    source = _patcher()
    first = source.get_model_object("first")
    mutation = ModelObjectPatchBatchMutation(
        (
            ExactModelObjectReplacement("first", first, nn.Identity()),
            ExactModelObjectReplacement("second", object(), nn.Identity()),
        )
    )

    with pytest.raises(ValueError, match="second.*expected object"):
        PATCHER_LIFECYCLE.derive_model(
            source,
            (mutation,),
            operation="atomic late identity regression",
        )

    assert source.object_patches == {}


@pytest.mark.parametrize("failure", [RuntimeError("failure"), KeyboardInterrupt()])
def test_batch_rolls_back_pending_state_on_exception_or_interruption(
    failure: BaseException,
) -> None:
    """Restore the exact pending map after any BaseException from Comfy's adder."""

    class FailingSurface:
        """Expose a valid exact surface whose second mutation fails."""

        def __init__(self) -> None:
            """Initialize two expected objects and one foreign pending patch."""

            self.values = {"first": object(), "second": object()}
            self.object_patches: dict[object, object] = {"foreign": object()}
            self.object_patches_backup: dict[object, object] = {}
            self.calls = 0

        def get_model_object(self, name: str) -> object:
            """Return one exact expected object."""

            return self.values[name]

        def add_object_patch(self, name: str, obj: object) -> None:
            """Install once, then emulate host failure or interruption."""

            self.calls += 1
            if self.calls == 2:
                raise failure
            self.object_patches[name] = obj

    model = FailingSurface()
    before = model.object_patches.copy()
    mutation = ModelObjectPatchBatchMutation(
        tuple(
            ExactModelObjectReplacement(path, model.values[path], object())
            for path in ("first", "second")
        )
    )

    with pytest.raises(type(failure)):
        mutation.apply(model)

    assert model.object_patches == before


def test_real_clone_patch_unpatch_and_reentry_restore_exact_graph() -> None:
    """Apply and eject one batch repeatedly without changing the source graph."""

    source = _patcher()
    first = source.get_model_object("first")
    second = source.get_model_object("second")
    first_replacement = nn.Identity()
    second_replacement = nn.Identity()
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        (
            ModelObjectPatchBatchMutation(
                (
                    ExactModelObjectReplacement("first", first, first_replacement),
                    ExactModelObjectReplacement("second", second, second_replacement),
                )
            ),
        ),
        operation="exact object reentry regression",
    )

    assert derived.parent is source
    assert source.object_patches == {}
    for _iteration in range(2):
        derived.patch_model(load_weights=False)
        assert derived.model.first is first_replacement
        assert derived.model.second is second_replacement
        derived.unpatch_model(unpatch_weights=False)
        assert derived.model.first is first
        assert derived.model.second is second
        assert derived.object_patches_backup == {}


def test_batch_rejects_ancestor_collision_before_mutation() -> None:
    """Prevent a module replacement from overlapping a nested active patch."""

    source = _patcher()
    derived = source.clone()
    derived.object_patches["first.weight"] = object()
    mutation = ModelObjectPatchBatchMutation(
        (
            ExactModelObjectReplacement(
                "first",
                source.get_model_object("first"),
                nn.Identity(),
            ),
        )
    )

    with pytest.raises(ValueError, match="collides with existing"):
        mutation.apply(derived)

    assert tuple(derived.object_patches) == ("first.weight",)


def test_batch_rejects_exact_collision_before_mutation() -> None:
    """Reject one exact pending path even when no distinct ancestor pair exists."""

    source = _patcher()
    derived = source.clone()
    existing = object()
    derived.object_patches["first"] = existing
    mutation = ModelObjectPatchBatchMutation(
        (
            ExactModelObjectReplacement(
                "first",
                source.get_model_object("first"),
                nn.Identity(),
            ),
        )
    )

    with pytest.raises(ValueError, match="collides with existing"):
        mutation.apply(derived)

    assert derived.object_patches == {"first": existing}


def _patcher() -> Any:
    """Return one real Comfy patcher around two exact module paths."""

    from comfy.model_patcher import ModelPatcher

    root = nn.Module()
    root.first = nn.Linear(2, 2)
    root.second = nn.Linear(2, 2)
    device = torch.device("cpu")
    return ModelPatcher(root, device, device)
