# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify CD owns complete local settings without separate tile dimensions."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.domain.sampler_options import (
    ContextualDiffusionOptions,
    TilingOptions,
)
from simple_syrup.nodes_v3.contextual_diffusion_options import (
    ContextualDiffusionOptionsV3,
)


def test_cd_schema_exposes_controls_without_sampling_payloads() -> None:
    """Keep context size authoritative while exposing all other spatial choices."""
    inputs = {
        value.id: value for value in ContextualDiffusionOptionsV3.define_schema().inputs
    }
    assert set(inputs) == {
        "options",
        "diffusion_mode",
        "latent_context_size",
        "latent_context_overlap",
        "latent_context_batch_size",
        "global_weight",
        "global_steps",
        "global_decay",
        "differential_diffusion",
    }
    assert inputs["options"].optional
    assert inputs["diffusion_mode"].default == "multidiffusion"
    assert inputs["latent_context_overlap"].default == 32
    assert inputs["latent_context_batch_size"].default == 4
    assert inputs["differential_diffusion"].default is False
    assert all(value.tooltip for value in inputs.values())


def test_existing_cd_widget_values_keep_their_serialized_positions() -> None:
    """Append new widgets so older workflows retain their context and global values."""
    inputs = ContextualDiffusionOptionsV3.define_schema().inputs
    widgets = [value.id for value in inputs if value.id not in {"options", "segs"}]
    assert widgets[:4] == [
        "latent_context_size",
        "global_weight",
        "global_steps",
        "global_decay",
    ]


def test_cd_defaults_preserve_its_existing_square_layout() -> None:
    """Retain the default CD-only recipe while allowing independent local edits."""
    (configured,) = ContextualDiffusionOptionsV3.execute()
    assert configured.contextual_diffusion == ContextualDiffusionOptions()
    assert configured.contextual_diffusion is not None
    assert configured.contextual_diffusion.local_tiling() == TilingOptions(
        width=96,
        height=96,
        overlap=32,
        batch_size=4,
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"diffusion_mode": "unknown"},
        {"latent_context_overlap": -1},
        {"latent_context_overlap": 96},
        {"latent_context_batch_size": 0},
        {"latent_context_size": 32, "latent_context_overlap": 32},
    ],
)
def test_cd_rejects_invalid_local_controls(controls: dict[str, Any]) -> None:
    """Fail before model execution when the chosen local plan is invalid."""
    with pytest.raises(ValueError):
        ContextualDiffusionOptionsV3.execute(**controls)


def test_cd_rejects_non_boolean_differential_diffusion() -> None:
    """Do not accept arbitrary workflow payloads for a mask-policy toggle."""
    malformed: Any = 1
    with pytest.raises(TypeError, match="boolean"):
        ContextualDiffusionOptionsV3.execute(differential_diffusion=malformed)
