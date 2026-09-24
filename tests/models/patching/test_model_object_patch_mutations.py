# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize concrete MODEL patcher mutation behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.runtime.model_patcher_mutations import (
    ModelExactObjectPatchMutation,
    ModelSharedObjectPatchMutation,
)


@pytest.mark.parametrize("path", ["", ".weight", "weight.", "model..weight"])
def test_exact_object_patch_rejects_invalid_dotted_paths(path: str) -> None:
    """Reject empty path segments before inspecting or changing patcher state."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="non-empty dotted path"):
        ModelExactObjectPatchMutation(path, object(), object()).apply(model)

    assert model.object_patches == {}


@pytest.mark.parametrize("state_name", ["object_patches", "object_patches_backup"])
def test_exact_object_patch_rejects_existing_active_or_backup_state(
    state_name: str,
) -> None:
    """Reject both forms of an already-owned exact object path."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected = model.get_model_object("weight")
    getattr(model, state_name)["weight"] = expected

    with pytest.raises(ValueError, match="already has a patch"):
        ModelExactObjectPatchMutation("weight", expected, object()).apply(model)

    assert (
        "weight" not in model.object_patches
        or model.object_patches["weight"] is expected
    )


def test_exact_object_patch_rejects_changed_current_identity() -> None:
    """Require the caller's exact expected object before claiming the path."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="does not match the expected object"):
        ModelExactObjectPatchMutation("weight", object(), object()).apply(model)

    assert model.object_patches == {}


def test_exact_object_patch_rejects_changed_adder_signature_before_lookup() -> None:
    """Validate the whole object-patch API before looking up or replacing an object."""

    class ChangedSurface:
        """Expose an incompatible object-patch adder."""

        def __init__(self) -> None:
            """Initialize valid state and untouched lookup state."""

            self.object_patches: dict[str, object] = {}
            self.object_patches_backup: dict[str, object] = {}
            self.looked_up = False

        def get_model_object(self, name: str) -> object:
            """Record a lookup that must never occur."""

            del name
            self.looked_up = True
            return object()

        def add_object_patch(self, path: str, value: object) -> None:
            """Expose deliberately changed parameter names."""

            del path, value

    model = ChangedSurface()

    with pytest.raises(TypeError, match="unsupported signature"):
        ModelExactObjectPatchMutation("weight", object(), object()).apply(model)

    assert model.looked_up is False
    assert model.object_patches == {}


@pytest.mark.parametrize("attribute_name", ["object_patches", "object_patches_backup"])
def test_exact_object_patch_rejects_malformed_state_dictionary(
    attribute_name: str,
) -> None:
    """Require both exact-path collision stores to remain dictionaries."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected = model.get_model_object("weight")
    setattr(model, attribute_name, None)

    with pytest.raises(TypeError, match=f"{attribute_name} must be a dictionary"):
        ModelExactObjectPatchMutation("weight", expected, object()).apply(model)

    setattr(model, attribute_name, {})


def test_shared_object_patch_accepts_exact_backup_and_live_replacement() -> None:
    """Register the already-live replacement when both identities remain exact."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected_backup = model.model.weight
    replacement = torch.nn.Parameter(torch.zeros_like(expected_backup))
    model.object_patches_backup["weight"] = expected_backup
    model.model.weight = replacement

    ModelSharedObjectPatchMutation(
        "weight",
        expected_backup,
        replacement,
    ).apply(model)

    assert model.object_patches["weight"] is replacement


@pytest.mark.parametrize("foreign_state", ["backup", "live"])
def test_shared_object_patch_rejects_foreign_shared_state(
    foreign_state: str,
) -> None:
    """Fail closed when either shared identity no longer belongs to the caller."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected_backup = model.model.weight
    replacement = torch.nn.Parameter(torch.zeros_like(expected_backup))
    model.object_patches_backup["weight"] = (
        object() if foreign_state == "backup" else expected_backup
    )
    model.model.weight = (
        torch.nn.Parameter(torch.ones_like(expected_backup))
        if foreign_state == "live"
        else replacement
    )

    expected_error = (
        "foreign shared backup" if foreign_state == "backup" else "foreign live"
    )
    with pytest.raises(ValueError, match=expected_error):
        ModelSharedObjectPatchMutation(
            "weight",
            expected_backup,
            replacement,
        ).apply(model)

    assert model.object_patches == {}


def _patcher(model: torch.nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
