# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the LayerStyle SAM models adapter node."""

from __future__ import annotations

import pytest

from simple_syrup.nodes.layerstyle_sam_models_adapter import LayerStyleSAMModelsAdapter


def test_layerstyle_adapter_contract() -> None:
    """Adapter splits LayerStyle bundles into conventional sockets."""

    assert LayerStyleSAMModelsAdapter.RETURN_TYPES == ("SAM_MODEL", "DINO_MODEL")
    assert LayerStyleSAMModelsAdapter.RETURN_NAMES == ("sam_model", "dino_model")
    assert LayerStyleSAMModelsAdapter.INPUT_TYPES()["required"]["sam_models"][0] == (
        "LS_SAM_MODELS"
    )


def test_layerstyle_adapter_returns_bundle_members() -> None:
    """Adapter returns SAM and DINO objects from the bundle."""

    sam = object()
    dino = object()

    result = LayerStyleSAMModelsAdapter().adapt({"SAM_MODEL": sam, "DINO_MODEL": dino})

    assert result == (sam, dino)


def test_layerstyle_adapter_rejects_invalid_bundle() -> None:
    """Invalid LayerStyle bundles fail clearly."""

    with pytest.raises(ValueError, match="SAM_MODEL and DINO_MODEL"):
        LayerStyleSAMModelsAdapter().adapt({"SAM_MODEL": object()})
