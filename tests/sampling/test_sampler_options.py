# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove graph-order independence, branch isolation, and capability conflicts."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import pytest

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.domain.sampler_options import (
    AttentionCouplingOptions,
    ContextualDiffusionOptions,
    SamplerCapability,
    SamplerOptions,
    TilingOptions,
    append_sampler_capability,
)


def test_all_distinct_capability_orders_produce_identical_configuration() -> None:
    """Let users arrange features freely without altering effective settings."""
    capabilities = (
        TilingOptions(),
        ContextualDiffusionOptions(),
        NoiseInversionOptions(),
        AttentionCouplingOptions(),
    )
    expected = SamplerOptions(
        tiling=capabilities[0],
        contextual_diffusion=capabilities[1],
        noise_inversion=capabilities[2],
        attention_coupling=capabilities[3],
    )
    all_capabilities: tuple[SamplerCapability, ...] = capabilities
    for order in permutations(all_capabilities):
        options: SamplerOptions | None = None
        for capability in order:
            options = append_sampler_capability(options, capability)
        assert options == expected


@pytest.mark.parametrize(
    "capability",
    [
        TilingOptions(),
        ContextualDiffusionOptions(),
        NoiseInversionOptions(),
        AttentionCouplingOptions(),
    ],
)
def test_every_feature_can_start_a_chain_and_rejects_duplicates(
    capability: SamplerCapability,
) -> None:
    """Avoid mandatory empty nodes and order-dependent last-wins overrides."""
    options = append_sampler_capability(None, capability)
    with pytest.raises(ValueError, match="Duplicate"):
        append_sampler_capability(options, capability)


def test_branched_options_do_not_mutate_the_shared_upstream_configuration() -> None:
    """Allow independent downstream samplers to share a configuration prefix."""
    upstream = SamplerOptions(tiling=TilingOptions())
    contextual = append_sampler_capability(upstream, ContextualDiffusionOptions())
    inverted = append_sampler_capability(upstream, NoiseInversionOptions())
    assert upstream.contextual_diffusion is None
    assert upstream.noise_inversion is None
    assert contextual.noise_inversion is None
    assert inverted.contextual_diffusion is None
    assert contextual.tiling is inverted.tiling is upstream.tiling


@pytest.mark.parametrize(
    ("constructor", "changes"),
    [
        (TilingOptions, {"width": 0}),
        (TilingOptions, {"overlap": 128}),
        (TilingOptions, {"diffusion_mode": "unknown"}),
        (TilingOptions, {"batch_size": 0}),
        (ContextualDiffusionOptions, {"global_weight": float("nan")}),
        (ContextualDiffusionOptions, {"global_decay": 2}),
        (ContextualDiffusionOptions, {"global_steps": -1}),
        (AttentionCouplingOptions, {"region_mask_feather": -1}),
        (
            AttentionCouplingOptions,
            {"regional_prompt_weight": 1.1},
        ),
    ],
)
def test_invalid_feature_settings_fail_at_construction(
    constructor: Any, changes: dict[str, Any]
) -> None:
    """Validate capability settings before any model mutation or sampling."""
    with pytest.raises(ValueError):
        constructor(**changes)


def test_malformed_options_connection_is_rejected() -> None:
    """Fail closed for foreign objects on the custom connection boundary."""
    malformed: Any = object()
    with pytest.raises(TypeError, match="Options input"):
        append_sampler_capability(malformed, TilingOptions())
    with pytest.raises(TypeError, match="tiling"):
        SamplerOptions(tiling=malformed)
