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
def test_inversion_controls_normalize_and_default_on(
    monkeypatch: pytest.MonkeyPatch,
    list_mode: bool,
) -> None:
    """Default to the accepted recipe and normalize zero-step disable in list mode."""
    monkeypatch.setattr(RecordingImplementation, "INPUT_IS_LIST", list_mode)
    text = ["prompt"] if list_mode else "prompt"
    output, enabled = RecordingAdapter.execute(text=text)
    assert output == text and enabled == NoiseInversionOptions()
    _, disabled = RecordingAdapter.execute(
        text=text,
        inversion_steps=[0] if list_mode else 0,
    )
    assert disabled is None


@pytest.mark.parametrize(
    "controls",
    [
        {"inversion_steps": -1},
        {"inversion_resolution_scale": 0},
        {"inversion_method": "fireflow"},
        {"inversion_switch_fraction": 1},
    ],
)
def test_invalid_selected_inversion_controls_fail(controls: dict[str, Any]) -> None:
    """Apply domain validation instead of handing malformed controls to a sampler."""
    with pytest.raises(ValueError):
        RecordingAdapter.execute(text="prompt", **controls)


@pytest.mark.parametrize(
    "node",
    [DetailSEGSAsRegionsV3, DetailSEGSByScaleFactorTiledDiffusionV3],
)
def test_existing_nodes_append_optional_inversion_without_reordering(node: Any) -> None:
    """Preserve sampler socket ordering before five optional inversion controls."""
    schema = node.define_schema()
    order = node.WORKFLOW_INPUT_ORDER
    assert [item.id for item in schema.inputs[: len(order)]] == list(order)
    new = schema.inputs[len(order) :]
    assert len(new) == 5 and all(item.optional and item.tooltip for item in new)
    assert new[0].id == "inversion_method" and new[0].default == "euler"
    steps = next(item for item in new if item.id == "inversion_steps")
    assert steps.default == 2 and steps.min == 0
