# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only MODEL modifier state snapshots."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tools.attention_coupling_benchmark.comfy_probe.model_modifier_snapshot import (
    SnapshotModelModifierV3,
    snapshot_model_modifier_state,
)


def _callback(*args: object, **kwargs: object) -> None:
    """Provide one stable callable identity for snapshot tests."""

    del args, kwargs


class _Holder:
    """Provide one recognizable core-cache holder type name."""


class _Model:
    """Expose the read-only patcher state consumed by the probe."""

    def __init__(self) -> None:
        """Create ordered modifier, wrapper, patch, and object state."""

        self.model_options = {
            "ppm_negpip": False,
            "model_function_wrapper": _callback,
            "transformer_options": {
                "easycache": _Holder(),
                "optimized_attention_override": _callback,
                "patches": {"attn2_patch": [_callback]},
            },
        }
        self.wrappers: dict[str, dict[object, list[Callable[..., object]]]] = {
            "outer_sample": {"easycache": [_callback]},
            "diffusion_model": {"upstream": [_callback, _callback]},
        }
        self.object_patches = {"diffusion_model.forward_orig": object()}


def test_snapshot_preserves_ordered_public_modifier_surfaces() -> None:
    """Record stable names and counts without serializing local objects."""

    model = _Model()

    snapshot = snapshot_model_modifier_state(model, run_id="run:case")

    assert snapshot == {
        "run_id": "run:case",
        "cache_holder_type": "_Holder",
        "model_function_wrapper": "_callback",
        "optimized_attention_override": "_callback",
        "ppm_negpip": False,
        "wrappers": [
            {
                "wrapper_type": "outer_sample",
                "key": "easycache",
                "callbacks": ["_callback"],
            },
            {
                "wrapper_type": "diffusion_model",
                "key": "upstream",
                "callbacks": ["_callback", "_callback"],
            },
        ],
        "object_patch_keys": ["diffusion_model.forward_orig"],
        "transformer_patch_counts": {"attn2_patch": 1},
    }
    transformer_options = model.model_options["transformer_options"]
    assert isinstance(transformer_options, dict)
    assert transformer_options["easycache"] is not None


def test_snapshot_node_passes_model_through_and_publishes_ui_evidence() -> None:
    """Keep the observation in the submitted graph without cloning MODEL state."""

    model = _Model()

    output = SnapshotModelModifierV3.execute(model, "run:case")

    assert output.result == (model,)
    assert output.ui["model_modifier_snapshot"] == [
        snapshot_model_modifier_state(model, run_id="run:case")
    ]
    schema = SnapshotModelModifierV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.SnapshotModelModifier"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("model_options", None, "model_options"),
        ("wrappers", None, "wrappers"),
        ("object_patches", None, "object_patches"),
    ],
)
def test_snapshot_rejects_malformed_patcher_containers(
    attribute: str,
    value: object,
    message: str,
) -> None:
    """Fail closed instead of omitting malformed modifier evidence."""

    model = _Model()
    setattr(model, attribute, value)

    with pytest.raises(TypeError, match=message):
        snapshot_model_modifier_state(model, run_id="run:case")
