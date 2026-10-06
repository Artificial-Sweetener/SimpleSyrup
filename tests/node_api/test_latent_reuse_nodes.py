# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify the public latent reuse node contracts and v3 registration."""

from typing import Any

import pytest

from simple_syrup.domain.sampler_options import (
    ContextualDiffusionOptions,
    SamplerOptions,
)
from simple_syrup.nodes_v3 import get_nodes
from simple_syrup.nodes_v3.latent_from_image import LatentFromImageV3
from simple_syrup.nodes_v3.noise_reuse_options import NoiseReuseOptionsV3
from simple_syrup.nodes_v3.simple_upscale_latent import SimpleUpscaleLatentV3
from simple_syrup.services.latent_provenance_service import LatentProvenanceService


@pytest.mark.parametrize(
    "node", (LatentFromImageV3, NoiseReuseOptionsV3, SimpleUpscaleLatentV3)
)
def test_v3_registration_and_tooltips_cover_public_contract(node: Any) -> None:
    """Export all three nodes with descriptions and actionable socket tooltips."""
    assert node in get_nodes()
    schema = node.define_schema()
    assert schema.node_id.startswith("SimpleSyrup.")
    assert schema.description
    assert all(value.tooltip for value in (*schema.inputs, *schema.outputs))


def test_upscale_exposes_one_dropdown_and_scale_default() -> None:
    """Keep format and model loading choices behind the agreed method control."""
    schema = SimpleUpscaleLatentV3.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert list(inputs) == ["latent", "method", "scale_factor"]
    assert inputs["method"].default == "Auto"
    assert "bislerp" in inputs["method"].tooltip.lower()
    assert inputs["scale_factor"].default == 2.0
    assert schema.display_name == "Simple Upscale Latent"


def test_reuse_enabled_is_first_widget_and_disabled_passes_same_prefix() -> None:
    """Prove zero finishing is reuse-only and switching off preserves prior options."""
    schema = NoiseReuseOptionsV3.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert list(inputs) == [
        "options",
        "enabled",
        "finishing_steps",
        "finishing_method",
        "finishing_start_fraction",
    ]
    assert inputs["enabled"].default is True
    assert inputs["finishing_steps"].default == 0
    prefix = SamplerOptions(contextual_diffusion=ContextualDiffusionOptions())
    assert NoiseReuseOptionsV3.execute(enabled=False, options=prefix)[0] is prefix
    enabled = NoiseReuseOptionsV3.execute(options=prefix)[0]
    assert enabled.contextual_diffusion is prefix.contextual_diffusion
    assert enabled.noise_reuse is not None and enabled.noise_reuse.finishing_steps == 0


def test_provenance_recovery_has_no_vae_or_widgets_and_never_encodes() -> None:
    """Return graph lineage directly instead of producing a re-encoded approximation."""
    schema = LatentFromImageV3.define_schema()
    assert [value.id for value in schema.inputs] == ["image"]
    service = LatentProvenanceService()
    prompt = {
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["loader", 2]},
        }
    }
    assert service.recover_link(prompt, ["decode", 0], {}) == ("sample", 0)
    with pytest.raises(ValueError, match="unmodified"):
        service.recover_link({}, ["loaded_image", 0], {})
