# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify capability execution routing, order independence and early rejection."""

from __future__ import annotations

from itertools import permutations, product
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from comfy.model_sampling import CONST

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.domain.sampler_options import (
    AttentionCouplingOptions,
    ContextualDiffusionOptions,
    SamplerCapability,
    SamplerOptions,
    TilingOptions,
)
from simple_syrup.nodes_v3.contextual_diffusion_options import (
    ContextualDiffusionOptionsV3,
)
from simple_syrup.nodes_v3.tiling_options import TilingOptionsV3
from simple_syrup.runtime import sampling_schedulers
from simple_syrup.services import sampler_options_sampling_service as routing


@pytest.fixture
def routed_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record sampling delegates while exercising the real configuration compiler."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def boundary(label: str) -> type:
        """Represent one expensive application sampling boundary."""

        class Recorder:
            """Keep neural execution outside configuration-routing unit tests."""

            def sample(self, **kwargs: Any) -> Any:
                """Retain complete routed values and the declared result shape."""
                calls.append((label, kwargs))
                result = kwargs["latent_image"]
                if label.startswith("context"):
                    return SimpleNamespace(latent=result, contexts=())
                return result

        return Recorder

    for attribute, label in (
        ("KSamplerSamplingService", "full"),
        ("AttentionCouplingSamplingService", "attention"),
        ("TiledDiffusionSamplingService", "tiled"),
        ("TiledAttentionCouplingSamplingService", "tiled_attention"),
        ("ContextualDiffusionSamplingService", "context"),
        ("ContextualAttentionCouplingSamplingService", "context_attention"),
    ):
        monkeypatch.setattr(routing, attribute, boundary(label))
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **kwargs: torch.tensor([0.5, 0.25, 0.0]),
    )
    return calls


def arguments() -> routing.SamplingArguments:
    """Use real flow-target validation with a small dynamic host MODEL boundary."""
    flow = CONST()
    flow.noise_scale = 1.0
    return {
        "model": SimpleNamespace(get_model_object=lambda name: flow),
        "seed": 17,
        "steps": 4,
        "cfg": 3.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": [],
        "negative": None,
        "latent_image": {"samples": torch.zeros((1, 4, 64, 96))},
        "denoise": 0.5,
    }


def capabilities() -> tuple[SamplerCapability, ...]:
    """Keep local geometry independent from the reduced whole-scene context."""
    return (
        TilingOptions(width=64, height=48, overlap=8, batch_size=2),
        ContextualDiffusionOptions(
            context_size=32, overlap=8, global_weight=0.8, global_steps=3
        ),
        NoiseInversionOptions(),
        AttentionCouplingOptions(0.7, 2),
    )


@pytest.mark.parametrize("enabled", tuple(product((False, True), repeat=4)))
@pytest.mark.parametrize("connected_segs", [None, object()])
@pytest.mark.parametrize("connected_masks", [None, object()])
def test_every_capability_combination_uses_one_sampling_authority(
    routed_calls: list[tuple[str, dict[str, Any]]],
    enabled: tuple[bool, bool, bool, bool],
    connected_segs: object | None,
    connected_masks: object | None,
) -> None:
    """Compose all 16 combinations without dropping configuration or negatives."""
    options = SamplerOptions()
    for capability, active in zip(capabilities(), enabled, strict=True):
        if active:
            options = options.with_capability(capability)
    values = arguments()
    if enabled[3] and connected_masks is not None:
        values["positive"] = ConditioningBatch(([], []))
    output = routing.SamplerOptionsSamplingService().sample(
        **values, options=options, segs=connected_segs, region_masks=connected_masks
    )
    assert output is values["latent_image"]
    assert len(routed_calls) == 1
    tile, context, inversion, attention = enabled
    attention = attention and connected_masks is not None
    expected = (
        ("context_attention" if attention else "context")
        if context
        else ("tiled_attention" if attention else "tiled")
        if tile
        else "attention"
        if attention
        else "full"
    )
    label, forwarded = routed_calls[0]
    assert label == expected
    if tile or context:
        assert forwarded["segs"] is connected_segs
    else:
        assert "segs" not in forwarded
    assert forwarded["negative"] is None
    assert forwarded["noise_inversion"] == (
        NoiseInversionOptions() if inversion else None
    )
    assert forwarded["model"] is values["model"]
    if context:
        assert forwarded["latent_context_size"] == 32
        assert forwarded["tiling"].width == 32
        assert forwarded["tiling"].height == 32
    elif tile:
        assert forwarded["latent_tile_width"] == 64
        assert forwarded["latent_tile_height"] == 48
    if attention:
        assert forwarded["region_masks"] is connected_masks
        assert forwarded["regional_prompt_weight"] == 0.7
        assert forwarded["region_mask_feather"] == 2
    else:
        assert "region_masks" not in forwarded


def test_all_connection_orders_compile_identical_execution(
    routed_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """Require all 24 capability orders to reach the same service with exact values."""
    values = arguments()
    values["positive"] = ConditioningBatch(([], []))
    masks = torch.ones((1, 64, 96))
    for order in permutations(capabilities()):
        options = SamplerOptions()
        for capability in order:
            options = options.with_capability(capability)
        output = routing.SamplerOptionsSamplingService().sample(
            **values, options=options, region_masks=masks
        )
        assert output is values["latent_image"]
        assert routed_calls[-1] == routed_calls[0]
    assert len(routed_calls) == 24


@pytest.mark.parametrize("options", [object(), {}, "MODEL"])
def test_foreign_options_fail_before_execution(
    routed_calls: list[tuple[str, dict[str, Any]]],
    options: Any,
) -> None:
    """Reject arbitrary socket payloads rather than interpreting them as controls."""
    with pytest.raises(TypeError, match="options nodes"):
        routing.SamplerOptionsSamplingService().sample(**arguments(), options=options)
    assert not routed_calls


def test_invalid_inversion_target_fails_before_attention_preparation(
    routed_calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a full-noise flow endpoint before any preparation or inference."""
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **kwargs: torch.tensor([1.0, 0.5, 0.0]),
    )
    options = SamplerOptions()
    for capability in capabilities():
        options = options.with_capability(capability)
    with pytest.raises(ValueError, match="full-noise endpoint"):
        routing.SamplerOptionsSamplingService().sample(**arguments(), options=options)
    assert not routed_calls


@pytest.mark.parametrize(
    "options",
    [
        SamplerOptions(tiling=TilingOptions()),
        SamplerOptions(contextual_diffusion=ContextualDiffusionOptions()),
    ],
)
def test_unipc_conflicts_fail_before_sampling(
    routed_calls: list[tuple[str, dict[str, Any]]],
    options: SamplerOptions,
) -> None:
    """Do not silently bypass tiling for an incompatible sampling method."""
    values = arguments()
    values["sampler_name"] = "uni_pc"
    with pytest.raises(ValueError, match="UniPC"):
        routing.SamplerOptionsSamplingService().sample(**values, options=options)
    assert not routed_calls


@pytest.mark.parametrize("cd_first", [False, True])
@pytest.mark.parametrize("attention", [False, True])
def test_cd_owns_all_local_controls_and_ignores_tiling_in_either_order(
    routed_calls: list[tuple[str, dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
    cd_first: bool,
    attention: bool,
) -> None:
    """Prove conflicting tiling cannot leak into CD routing, including its preflight."""
    cd_segs = object()
    (cd,) = ContextualDiffusionOptionsV3.execute(
        diffusion_mode="mixture_of_diffusers",
        latent_context_size=48,
        latent_context_overlap=12,
        latent_context_batch_size=2,
        differential_diffusion=True,
    )
    base = cd
    if attention:
        base = base.with_capability(AttentionCouplingOptions())
    values = arguments()
    masks = torch.ones((1, 64, 96)) if attention else None
    if attention:
        values["positive"] = ConditioningBatch(([], []))
    routing.SamplerOptionsSamplingService().sample(
        **values, options=base, segs=cd_segs, region_masks=masks
    )
    reference = routed_calls[-1]
    assert not caplog.records

    def tiling_node(options: SamplerOptions | None = None) -> SamplerOptions:
        """Build a conflicting local configuration at the actual node boundary."""
        (result,) = TilingOptionsV3.execute(
            options=options,
            diffusion_mode="multidiffusion",
            latent_tile_width=128,
            latent_tile_height=64,
            latent_tile_overlap=32,
            latent_tile_batch_size=4,
            differential_diffusion=False,
        )
        return result

    if cd_first:
        combined = tiling_node(base)
    else:
        tiles = tiling_node()
        assert base.contextual_diffusion is not None
        combined = tiles.with_capability(base.contextual_diffusion)
        if attention:
            assert base.attention_coupling is not None
            combined = combined.with_capability(base.attention_coupling)
    routing.SamplerOptionsSamplingService().sample(
        **values, options=combined, segs=cd_segs, region_masks=masks
    )
    assert routed_calls[-1] == reference
    forwarded = reference[1]
    assert forwarded["tiling"] == TilingOptions(
        diffusion_mode="mixture_of_diffusers",
        width=48,
        height=48,
        overlap=12,
        batch_size=2,
        differential_diffusion=True,
    )
    assert forwarded["segs"] is cd_segs
    assert forwarded["latent_context_overlap"] == 12
    assert forwarded["latent_context_batch_size"] == 2
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].getMessage() == (
        "Contextual Diffusion takes precedence; Tiling Options ignored."
    )
    assert caplog.records[0].__dict__["node_id"] == "SimpleSyrup.KSampler"
    # Bypassing CD leaves the original tiling capability intact.
    assert combined.tiling is not None
    routing.SamplerOptionsSamplingService().sample(
        **values, options=SamplerOptions(tiling=combined.tiling), segs=cd_segs
    )
    assert routed_calls[-1][0] == "tiled"
    assert routed_calls[-1][1]["latent_tile_width"] == 128
    assert routed_calls[-1][1]["segs"] is cd_segs
    assert len(caplog.records) == 1


def test_inversion_preflight_uses_cd_geometry_not_ignored_tiles(
    routed_calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the effective CD view when validating geometry-dependent inversion sigmas."""
    views: list[sampling_schedulers.SchedulerView] = []

    def sigmas(**kwargs: Any) -> torch.Tensor:
        """Observe the external scheduler boundary without running model inference."""
        views.append(kwargs["view"])
        return torch.tensor([0.5, 0.25, 0.0])

    monkeypatch.setattr(sampling_schedulers, "calculate_sigmas", sigmas)
    routing.SamplerOptionsSamplingService().sample(
        **arguments(),
        options=SamplerOptions(
            tiling=TilingOptions(width=128, height=64),
            contextual_diffusion=ContextualDiffusionOptions(context_size=48, overlap=8),
            noise_inversion=NoiseInversionOptions(),
        ),
    )
    assert views == [sampling_schedulers.SchedulerView(48, 48)]
    assert len(routed_calls) == 1
