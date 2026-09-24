# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove safe regional LoRA prepared-weight cache identity and reuse."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterIdentity
from simple_syrup.runtime.regional_lora.execution_cache import (
    REGIONAL_LORA_EXECUTION_CACHE_KEY_FACTORY,
    ModelCloneLineage,
    RegionalLoraExecutionCache,
    RegionalLoraExecutionCacheKey,
    RegionalLoraPreparedWeights,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_equal_effective_clone_lineage_and_content_reuse_one_preparation() -> None:
    """Reuse prepared weights across descendants with identical global patches."""

    source = _patcher(torch.nn.Linear(3, 4))
    clone = source.clone()
    adapter = _adapter()
    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    source_key = _key(identity, source, adapter)
    clone_key = _key(identity, clone, adapter)
    cache = RegionalLoraExecutionCache()
    calls = 0

    def prepare() -> RegionalLoraPreparedWeights:
        """Prepare one independently owned tensor pair."""

        nonlocal calls
        calls += 1
        return RegionalLoraPreparedWeights(
            down=adapter.down.clone(),
            up=adapter.up.clone(),
        )

    first = cache.get_or_prepare(source_key, prepare)
    second = cache.get_or_prepare(clone_key, prepare)

    assert source_key == clone_key
    assert first is second
    assert calls == 1
    assert cache.size == 1


def test_every_required_cache_dimension_changes_identity() -> None:
    """Separate adapter, model, target, device, dtype, and patch-content changes."""

    model = _patcher(torch.nn.Linear(3, 4))
    adapter = _adapter()
    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    baseline = _key(identity, model, adapter)
    other_model = _patcher(torch.nn.Linear(3, 4))
    other_target = StandardLoraTarget(
        target="diffusion_model.blocks.1.self_attn.q_proj",
        down=adapter.down,
        up=adapter.up,
        rank=adapter.rank,
        input_features=adapter.input_features,
        output_features=adapter.output_features,
    )
    changed_down = adapter.down.clone()
    changed_down[0, 0] += 1.0
    changed_content = StandardLoraTarget(
        target=adapter.target,
        down=changed_down,
        up=adapter.up,
        rank=adapter.rank,
        input_features=adapter.input_features,
        output_features=adapter.output_features,
    )

    variants = (
        _key(RegionalLoraAdapterIdentity("other.safetensors"), model, adapter),
        _key(identity, other_model, adapter),
        _key(identity, model, other_target),
        _key(identity, model, adapter, device="cuda:0"),
        _key(identity, model, adapter, dtype=torch.float16),
        _key(identity, model, changed_content),
    )

    assert all(variant != baseline for variant in variants)
    assert len(set(variants)) == len(variants)


def test_changed_global_patch_state_separates_same_clone_family() -> None:
    """Include effective global patches while retaining clone-family reuse."""

    source = _patcher(torch.nn.Linear(3, 4))
    clone = source.clone()
    adapter = _adapter()
    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    before = _key(identity, clone, adapter)
    clone.patches_uuid = uuid4()
    after = _key(identity, clone, adapter)

    assert clone.clone_base_uuid == source.clone_base_uuid
    assert before.model_lineage.clone_base_uuid == after.model_lineage.clone_base_uuid
    assert before.model_lineage.patches_uuid != after.model_lineage.patches_uuid
    assert before != after


def test_patch_content_digest_changes_after_source_tensor_mutation() -> None:
    """Prevent stale hits when caller-owned admitted tensor content changes."""

    model = _patcher(torch.nn.Linear(3, 4))
    adapter = _adapter()
    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    before = _key(identity, model, adapter)

    adapter.down[0, 0] += 1.0
    after = _key(identity, model, adapter)

    assert before.patch_content != after.patch_content


def test_cache_contract_excludes_schedule_and_spatial_state() -> None:
    """Make stale schedule or mask capture impossible in key and value schemas."""

    field_names = {field.name for field in fields(RegionalLoraExecutionCacheKey)} | {
        field.name for field in fields(RegionalLoraPreparedWeights)
    }
    forbidden_fragments = (
        "schedule",
        "strength",
        "mask",
        "region",
        "spatial",
        "view",
        "query",
    )

    assert all(
        fragment not in field_name
        for field_name in field_names
        for fragment in forbidden_fragments
    )


@pytest.mark.parametrize(
    ("prepared", "message"),
    [
        (
            RegionalLoraPreparedWeights(
                torch.ones((2, 3), dtype=torch.float16),
                torch.ones((4, 2), dtype=torch.float16),
            ),
            "wrong dtype",
        ),
        (
            RegionalLoraPreparedWeights(
                torch.ones((2, 4)),
                torch.ones((4, 2)),
            ),
            "down tensor has the wrong shape",
        ),
        (
            RegionalLoraPreparedWeights(
                torch.ones((2, 3)),
                torch.ones((5, 2)),
            ),
            "up tensor has the wrong shape",
        ),
    ],
)
def test_cache_rejects_malformed_prepared_values_without_storing(
    prepared: RegionalLoraPreparedWeights,
    message: str,
) -> None:
    """Reject mismatched residency and shapes without poisoning the cache."""

    cache = RegionalLoraExecutionCache()
    key = _key(
        RegionalLoraAdapterIdentity("adapter.safetensors"),
        _patcher(torch.nn.Linear(3, 4)),
        _adapter(),
    )

    with pytest.raises(ValueError, match=message):
        cache.get_or_prepare(key, lambda: prepared)

    assert cache.size == 0


def _adapter() -> StandardLoraTarget:
    """Return one recognizable admitted CPU adapter target."""

    return StandardLoraTarget(
        target="diffusion_model.blocks.0.self_attn.q_proj",
        down=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        up=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        rank=2,
        input_features=3,
        output_features=4,
    )


def _key(
    identity: RegionalLoraAdapterIdentity,
    model: object,
    adapter: StandardLoraTarget,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RegionalLoraExecutionCacheKey:
    """Build one key through the production identity owner."""

    return REGIONAL_LORA_EXECUTION_CACHE_KEY_FACTORY.build(
        adapter_identity=identity,
        model_lineage=ModelCloneLineage.from_model(model),
        adapter=adapter,
        device=device,
        dtype=dtype,
    )


def _patcher(model: torch.nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
