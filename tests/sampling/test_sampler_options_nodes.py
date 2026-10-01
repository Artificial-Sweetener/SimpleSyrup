# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify native option sockets, defaults, validation and graph composition."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import pytest
import torch

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.domain.sampler_options import SamplerOptions
from simple_syrup.nodes_v3.attention_coupling_options import AttentionCouplingOptionsV3
from simple_syrup.nodes_v3.contextual_diffusion_options import (
    ContextualDiffusionOptionsV3,
)
from simple_syrup.nodes_v3.ksampler import KSamplerV3
from simple_syrup.nodes_v3.ksampler_contextual_diffusion import (
    KSamplerContextualDiffusionV3,
)
from simple_syrup.nodes_v3.noise_inversion_options import NoiseInversionOptionsV3
from simple_syrup.nodes_v3.sampler_options_schema import OPTIONS_TYPE
from simple_syrup.nodes_v3.tiling_options import TilingOptionsV3
from simple_syrup.services.sampler_options_sampling_service import SamplingArguments

OPTION_NODES: tuple[Any, ...] = (
    TilingOptionsV3,
    ContextualDiffusionOptionsV3,
    NoiseInversionOptionsV3,
    AttentionCouplingOptionsV3,
)


@pytest.mark.parametrize("node", OPTION_NODES)
def test_each_capability_can_start_a_natively_bypassable_chain(node: Any) -> None:
    """Use one matching optional input/output, without an empty or enable node."""
    schema = node.define_schema()
    options = next(value for value in schema.inputs if value.id == "options")
    assert options.optional and options.io_type == OPTIONS_TYPE
    assert len(schema.outputs) == 1
    assert schema.outputs[0].io_type == OPTIONS_TYPE
    assert "enabled" not in " ".join(value.id for value in schema.inputs)
    assert schema.description
    assert all(value.tooltip for value in (*schema.inputs, *schema.outputs))


def test_all_node_orders_preserve_configuration_and_branches() -> None:
    """Build all 24 real node chains without touching MODEL preparation."""
    masks = torch.ones((1, 32, 64))
    reference: SamplerOptions | None = None
    for order in permutations(OPTION_NODES):
        chain: SamplerOptions | None = None
        for node in order:
            kwargs: dict[str, Any] = {"options": chain}
            if node is AttentionCouplingOptionsV3:
                kwargs["region_masks"] = masks
            previous = chain
            (chain,) = node.execute(**kwargs)
            assert chain is not previous
        assert isinstance(chain, SamplerOptions)
        if reference is None:
            reference = chain
        else:
            assert chain == reference
    assert reference is not None
    assert reference.noise_inversion == NoiseInversionOptions()


@pytest.mark.parametrize("node", OPTION_NODES)
def test_duplicate_capability_node_fails_explicitly(node: Any) -> None:
    """Reject ambiguous controls rather than choosing a winner by connection order."""
    kwargs = (
        {"region_masks": torch.ones((1, 4, 4))}
        if node is AttentionCouplingOptionsV3
        else {}
    )
    (first,) = node.execute(**kwargs)
    with pytest.raises(ValueError, match="Duplicate sampler capability"):
        node.execute(options=first, **kwargs)


def test_inversion_node_defaults_match_convenience_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure both entry points construct the same accepted inversion configuration."""
    calls: list[dict[str, Any]] = []

    class RecordingService:
        """Capture the node-to-service boundary without model execution."""

        def sample(self, **kwargs: Any) -> Any:
            """Expose only a stable Comfy-facing result shape."""
            from types import SimpleNamespace

            calls.append(kwargs)
            return SimpleNamespace(latent=kwargs["latent_image"], contexts=object())

    monkeypatch.setattr(
        KSamplerContextualDiffusionV3, "service_class", RecordingService
    )
    (options,) = NoiseInversionOptionsV3.execute()
    standard: SamplingArguments = {
        "model": object(),
        "seed": 1,
        "steps": 4,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": [],
        "negative": None,
        "latent_image": {"samples": torch.zeros((1, 4, 32, 64))},
        "denoise": 0.5,
    }
    KSamplerContextualDiffusionV3.execute(**standard, noise_inversion_enabled=True)
    assert (
        calls[-1]["noise_inversion"]
        == options.noise_inversion
        == NoiseInversionOptions()
    )
    KSamplerContextualDiffusionV3.execute(**standard)
    assert calls[-1]["noise_inversion"] is None
    assert calls[-1]["negative"] is None


def test_new_sampler_preserves_optional_negative_and_consumes_typed_options() -> None:
    """Keep standard Comfy sampling sockets with one optional capability connection."""
    inputs = {value.id: value for value in KSamplerV3.define_schema().inputs}
    assert inputs["negative"].optional
    assert inputs["options"].optional and inputs["options"].io_type == OPTIONS_TYPE
    assert inputs["steps"].default == 20 and inputs["cfg"].default == 8.0
