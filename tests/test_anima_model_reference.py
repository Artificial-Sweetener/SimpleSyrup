# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify non-owning Anima wrapper identity and patch geometry."""

from __future__ import annotations

import gc
import weakref

import pytest
from torch import nn

from simple_syrup.runtime.regional_lora.anima_model_reference import (
    AnimaDiffusionModelReference,
)


class _PatchModel(nn.Module):
    """Expose the immutable patch geometry required by Anima wrappers."""

    patch_temporal = 1
    patch_spatial = 2


def test_reference_reads_geometry_without_retaining_model() -> None:
    """A released model must not survive solely through wrapper identity."""

    model = _PatchModel()
    model_reference = weakref.ref(model)
    reference = AnimaDiffusionModelReference(model)

    assert reference.owns(model)
    assert reference.patch_temporal == 1
    assert reference.patch_spatial == 2

    del model
    gc.collect()

    assert model_reference() is None
    assert not reference.owns(object())


@pytest.mark.parametrize(
    "attribute,value", [("patch_temporal", 0), ("patch_spatial", 1.5)]
)
def test_reference_rejects_invalid_patch_geometry(
    attribute: str,
    value: object,
) -> None:
    """Invalid installed patch geometry fails before any wrapper is installed."""

    model = _PatchModel()
    setattr(model, attribute, value)

    reference = AnimaDiffusionModelReference(model)

    with pytest.raises(ValueError, match=attribute):
        getattr(reference, attribute)
