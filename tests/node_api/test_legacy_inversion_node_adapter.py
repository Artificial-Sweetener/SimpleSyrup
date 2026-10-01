# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify optional inversion widgets on implementation-backed V3 sampler nodes."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.nodes_v3.legacy_inversion_node_adapter import (
    LegacyInversionNodeV3Adapter,
)
from simple_syrup.nodes_v3.legacy_node_wrappers import (
    DetailSEGSAsRegionsV3,
    DetailSEGSByScaleFactorTiledDiffusionV3,
    KSamplerExtrasV3,
)


class RecordingImplementation:
    """Expose a minimal maintained declaration and the execution boundary."""

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Result.",)
    FUNCTION = "run"
    CATEGORY = "SimpleSyrup/Test"
    DESCRIPTION = "Records sampling inputs."
    INPUT_IS_LIST = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """Keep the persisted input contract independent of inversion widgets."""
        return {"required": {"text": ("STRING", {"default": "", "tooltip": "Text."})}}

    def run(
        self, text: object, noise_inversion: NoiseInversionOptions | None = None
    ) -> tuple[object, NoiseInversionOptions | None]:
        """Return delegated inputs without invoking Comfy neural execution."""
        return text, noise_inversion


class RecordingAdapter(LegacyInversionNodeV3Adapter):
    """Use the production schema and list-mode normalization implementation."""

    LEGACY_NODE_CLASS: ClassVar[type[Any]] = RecordingImplementation
    NODE_ID = "SimpleSyrup.Recording"
    DISPLAY_NAME = "Recording"


@pytest.mark.parametrize("list_mode", [False, True])
def test_inversion_controls_normalize_and_default_off(
    monkeypatch: pytest.MonkeyPatch,
    list_mode: bool,
) -> None:
    """Preserve existing execution while constructing the accepted recipe on request."""
    monkeypatch.setattr(RecordingImplementation, "INPUT_IS_LIST", list_mode)
    text = ["prompt"] if list_mode else "prompt"
    output, disabled = RecordingAdapter.execute(text=text)
    assert output == text and disabled is None
    _, enabled = RecordingAdapter.execute(
        text=text,
        noise_inversion_enabled=[True] if list_mode else True,
    )
    assert enabled == NoiseInversionOptions()


@pytest.mark.parametrize(
    "controls",
    [
        {"inversion_steps": 0},
        {"inversion_resolution_scale": 0},
        {"inversion_method": "fireflow"},
        {"inversion_switch_fraction": 1},
    ],
)
def test_invalid_selected_inversion_controls_fail(controls: dict[str, Any]) -> None:
    """Apply domain validation instead of handing malformed controls to a sampler."""
    with pytest.raises(ValueError):
        RecordingAdapter.execute(
            text="prompt", noise_inversion_enabled=True, **controls
        )


@pytest.mark.parametrize(
    "node",
    [KSamplerExtrasV3, DetailSEGSAsRegionsV3, DetailSEGSByScaleFactorTiledDiffusionV3],
)
def test_existing_nodes_append_optional_inversion_without_reordering(node: Any) -> None:
    """Preserve every serialized socket position before seven optional new controls."""
    schema = node.define_schema()
    order = node.WORKFLOW_INPUT_ORDER
    assert [item.id for item in schema.inputs[: len(order)]] == list(order)
    new = schema.inputs[len(order) :]
    assert len(new) == 7 and all(item.optional and item.tooltip for item in new)
    assert new[0].id == "noise_inversion_enabled" and new[0].default is False
