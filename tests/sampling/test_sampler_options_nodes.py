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
from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_contextual_attention_coupling import (
    KSamplerContextualAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_contextual_diffusion import (
    KSamplerContextualDiffusionV3,
)
from simple_syrup.nodes_v3.ksampler_prompt_by_region import KSamplerPromptByRegionV3
from simple_syrup.nodes_v3.ksampler_prompt_by_tiled_region import (
    KSamplerPromptByTiledRegionV3,
)
from simple_syrup.nodes_v3.ksampler_tiled_attention_coupling import (
    KSamplerTiledAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_tiled_diffusion import KSamplerTiledDiffusionV3
from simple_syrup.nodes_v3.noise_inversion_options import NoiseInversionOptionsV3
from simple_syrup.nodes_v3.sampler_options_schema import (
    OPTIONS_TYPE,
    inversion_from_controls,
)
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
    assert "segs" not in {value.id for value in schema.inputs}
    assert "region_masks" not in {value.id for value in schema.inputs}


def test_all_node_orders_preserve_configuration_and_branches() -> None:
    """Build all 24 real node chains without touching MODEL preparation."""
    reference: SamplerOptions | None = None
    for order in permutations(OPTION_NODES):
        chain: SamplerOptions | None = None
        for node in order:
            kwargs: dict[str, Any] = {"options": chain}
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
    (first,) = node.execute()
    with pytest.raises(ValueError, match="Duplicate sampler capability"):
        node.execute(options=first)


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
    KSamplerContextualDiffusionV3.execute(**standard)
    assert (
        calls[-1]["noise_inversion"]
        == options.noise_inversion
        == NoiseInversionOptions()
    )
    KSamplerContextualDiffusionV3.execute(**standard, inversion_steps=0)
    assert calls[-1]["noise_inversion"] is None
    assert calls[-1]["negative"] is None


def test_zero_inversion_steps_pass_through_existing_options() -> None:
    """Disable both inversion stages without discarding preceding capabilities."""
    (upstream,) = TilingOptionsV3.execute()
    (result,) = NoiseInversionOptionsV3.execute(options=upstream, inversion_steps=0)
    assert result is upstream
    assert result.noise_inversion is None
    (empty,) = NoiseInversionOptionsV3.execute(inversion_steps=0)
    assert empty == SamplerOptions()
    assert (
        inversion_from_controls(inversion_steps=0, inversion_finishing_steps=8) is None
    )


@pytest.mark.parametrize("steps", [-1, 65, True, 1.5])
def test_inversion_step_disable_rejects_invalid_counts(steps: Any) -> None:
    """Reserve integer zero for disabling rather than accepting false-like inputs."""
    with pytest.raises(ValueError, match="steps"):
        inversion_from_controls(inversion_steps=steps)


def test_inversion_schema_has_one_method_and_zero_step_disable() -> None:
    """Keep the accepted default enabled through five unambiguous controls."""
    inputs = {
        value.id: value for value in NoiseInversionOptionsV3.define_schema().inputs
    }
    assert "noise_inversion_enabled" not in inputs
    assert "inversion_finishing_method" not in inputs
    assert inputs["inversion_steps"].default == 2
    assert inputs["inversion_steps"].min == 0


@pytest.mark.parametrize(
    "node",
    [
        KSamplerAttentionCouplingV3,
        KSamplerContextualAttentionCouplingV3,
        KSamplerContextualDiffusionV3,
        KSamplerPromptByRegionV3,
        KSamplerPromptByTiledRegionV3,
        KSamplerTiledAttentionCouplingV3,
        KSamplerTiledDiffusionV3,
    ],
)
@pytest.mark.parametrize("inversion_steps", [0, 2])
def test_every_native_sampler_uses_shared_default_and_disable(
    node: Any, inversion_steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise all sampler entry points with enabled defaults and explicit disable."""
    from types import SimpleNamespace

    calls: list[dict[str, Any]] = []

    class RecordingService:
        """Observe orchestration without invoking external model execution."""

        def sample(self, **kwargs: Any) -> Any:
            """Return either the contextual result or the normal latent shape."""
            calls.append(kwargs)
            if "latent_context_size" in kwargs:
                return SimpleNamespace(latent=kwargs["latent_image"], contexts=object())
            return kwargs["latent_image"]

        def assemble(self, **kwargs: Any) -> tuple[object, object]:
            """Pass through regional conditioning at its external boundary."""
            return kwargs["positive"], kwargs["negative"]

    for name in (
        "service_class",
        "sampling_service_class",
        "conditioning_service_class",
    ):
        if hasattr(node, name):
            monkeypatch.setattr(node, name, RecordingService)
    controls: dict[str, Any] = {} if inversion_steps == 2 else {"inversion_steps": 0}
    node.execute(
        model=object(),
        seed=1,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=[],
        negative=None,
        latent_image={"samples": torch.zeros((1, 4, 8, 8))},
        region_masks=torch.ones((1, 8, 8)),
        denoise=0.5,
        **controls,
    )
    assert calls[-1]["noise_inversion"] == (
        NoiseInversionOptions() if inversion_steps else None
    )
    inputs = {value.id: value for value in node.define_schema().inputs}
    assert inputs["inversion_steps"].default == 2
    assert inputs["inversion_steps"].min == 0
    assert "noise_inversion_enabled" not in inputs
    assert "inversion_finishing_method" not in inputs


def test_new_sampler_preserves_optional_negative_and_consumes_typed_options() -> None:
    """Keep standard Comfy sampling sockets with one optional capability connection."""
    inputs = {value.id: value for value in KSamplerV3.define_schema().inputs}
    assert inputs["negative"].optional
    assert inputs["segs"].optional and inputs["segs"].io_type == "SEGS"
    assert inputs["region_masks"].optional and inputs["region_masks"].io_type == "MASK"
    assert inputs["options"].optional and inputs["options"].io_type == OPTIONS_TYPE
    assert inputs["steps"].default == 20 and inputs["cfg"].default == 8.0


def test_sampler_forwards_region_data_without_mutating_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep region data on the execution boundary rather than the reusable chain."""
    calls: list[dict[str, Any]] = []

    class RecordingService:
        """Observe the sampler's public delegation without neural execution."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Return the exact latent payload after recording connected regions."""
            calls.append(kwargs)
            result: dict[str, Any] = kwargs["latent_image"]
            return result

    monkeypatch.setattr(KSamplerV3, "service_class", RecordingService)
    (options,) = TilingOptionsV3.execute()
    segs = object()
    masks = torch.ones((1, 16, 16))
    latent = {"samples": torch.zeros((1, 4, 16, 16))}
    assert KSamplerV3.execute(
        model=object(),
        seed=17,
        steps=4,
        cfg=3.0,
        sampler_name="euler",
        scheduler="simple",
        positive=[],
        latent_image=latent,
        options=options,
        segs=segs,
        region_masks=masks,
    ) == (latent,)
    assert calls[0]["segs"] is segs
    assert calls[0]["region_masks"] is masks
    assert calls[0]["options"] is options
    assert calls[0]["negative"] is None
