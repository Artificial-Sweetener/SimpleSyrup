# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the benchmark sampler schema matches the production boundary."""

from __future__ import annotations

import asyncio

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)
from tools.attention_coupling_benchmark.comfy_probe import (
    attention_coupling_phase_node,
    comfy_entrypoint,
)

ProfiledKSamplerAttentionCouplingV3 = (
    attention_coupling_phase_node.ProfiledKSamplerAttentionCouplingV3
)


def test_profiled_sampler_preserves_input_and_output_schema() -> None:
    """Change only benchmark identity and dev-only visibility."""

    production = KSamplerAttentionCouplingV3.define_schema()
    profiled = ProfiledKSamplerAttentionCouplingV3.define_schema()

    assert profiled.node_id == "SimpleSyrupBenchmark.ProfiledKSamplerAttentionCoupling"
    assert profiled.is_dev_only is True
    assert [value.id for value in profiled.inputs] == [
        value.id for value in production.inputs
    ]
    assert [value.io_type for value in profiled.inputs] == [
        value.io_type for value in production.inputs
    ]
    assert [value.io_type for value in profiled.outputs] == [
        value.io_type for value in production.outputs
    ]


def test_profiled_sampler_is_registered_only_by_benchmark_extension() -> None:
    """Make the diagnostic graph executable without changing product exports."""

    extension = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert ProfiledKSamplerAttentionCouplingV3 in nodes


def test_profiled_sampler_bridges_host_conditioning_batches_before_delegation() -> None:
    """Normalize only canonical host values at the sibling-extension boundary."""

    calls: list[dict[str, object]] = []

    class _Service:
        """Capture inherited-node arguments and return the supplied latent."""

        def sample(self, **arguments: object) -> dict[str, object]:
            """Retain exact arguments for namespace assertions."""

            calls.append(arguments)
            latent = arguments["latent_image"]
            assert isinstance(latent, dict)
            return latent

    host_type = type(
        "ConditioningBatch",
        (),
        {
            "__module__": (
                "custom_nodes.SimpleSyrup.simple_syrup.domain.conditioning_batch"
            )
        },
    )
    positive = host_type()
    positive.entries = ("positive",)
    negative = host_type()
    negative.entries = ("negative",)
    original = ProfiledKSamplerAttentionCouplingV3.sampling_service_class
    ProfiledKSamplerAttentionCouplingV3.sampling_service_class = _Service  # type: ignore[assignment]
    latent = {"samples": torch.zeros((1, 4, 2, 2))}
    try:
        output = ProfiledKSamplerAttentionCouplingV3.execute(
            model=object(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="sampler",
            scheduler="scheduler",
            positive=positive,
            negative=negative,
            region_masks=object(),
            regional_prompt_weight=1.0,
            region_mask_feather=0,
            latent_image=latent,
            denoise=1.0,
        )
    finally:
        ProfiledKSamplerAttentionCouplingV3.sampling_service_class = original

    assert output == (latent,)
    assert isinstance(calls[0]["positive"], ConditioningBatch)
    assert isinstance(calls[0]["negative"], ConditioningBatch)
