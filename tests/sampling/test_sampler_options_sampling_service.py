"""Verify capability execution routing, order independence and early rejection."""

from __future__ import annotations

from itertools import permutations, product
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from comfy.model_sampling import CONST

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.domain.sampler_options import (
    AttentionCouplingOptions,
    ContextualDiffusionOptions,
    SamplerCapability,
    SamplerOptions,
    TilingOptions,
)
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
        ContextualDiffusionOptions(context_size=32, global_weight=0.8, global_steps=3),
        NoiseInversionOptions(),
        AttentionCouplingOptions(torch.ones((1, 64, 96)), 0.7, 2),
    )


@pytest.mark.parametrize("enabled", tuple(product((False, True), repeat=4)))
def test_every_capability_combination_uses_one_sampling_authority(
    routed_calls: list[tuple[str, dict[str, Any]]],
    enabled: tuple[bool, bool, bool, bool],
) -> None:
    """Compose all 16 combinations without dropping configuration or negatives."""
    options = SamplerOptions()
    for capability, active in zip(capabilities(), enabled, strict=True):
        if active:
            options = options.with_capability(capability)
    values = arguments()
    output = routing.SamplerOptionsSamplingService().sample(**values, options=options)
    assert output is values["latent_image"]
    assert len(routed_calls) == 1
    tile, context, inversion, attention = enabled
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
    assert forwarded["negative"] is None
    assert forwarded["noise_inversion"] == (
        NoiseInversionOptions() if inversion else None
    )
    assert forwarded["model"] is values["model"]
    if context:
        assert forwarded["latent_context_size"] == 32
        assert forwarded["tiling"].width == (64 if tile else 32)
        assert forwarded["tiling"].height == (48 if tile else 32)
    elif tile:
        assert forwarded["latent_tile_width"] == 64
        assert forwarded["latent_tile_height"] == 48
    if attention:
        assert forwarded["regional_prompt_weight"] == 0.7
        assert forwarded["region_mask_feather"] == 2


def test_all_connection_orders_compile_identical_execution(
    routed_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """Require all 24 capability orders to reach the same service with exact values."""
    values = arguments()
    for order in permutations(capabilities()):
        options = SamplerOptions()
        for capability in order:
            options = options.with_capability(capability)
        output = routing.SamplerOptionsSamplingService().sample(
            **values, options=options
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
