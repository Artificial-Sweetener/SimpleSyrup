# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact-input prepared Attention Coupling result reuse."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.anima_attention_coupling_model_family import (
    AnimaAttentionCouplingModelFamily,
)
from simple_syrup.services.attention_coupling_model_family import (
    AttentionCouplingPreparedModelReuse,
)
from simple_syrup.services.attention_coupling_prepared_model_cache import (
    AttentionCouplingPreparedModelCache,
    AttentionCouplingPreparedRequest,
)
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)
from simple_syrup.services.unet_attention_coupling_model_family import (
    StandardUnetAttentionCouplingModelFamily,
)


def test_exact_request_returns_the_same_prepared_object() -> None:
    """Publish one miss and reuse its exact prepared object identity."""

    cache = AttentionCouplingPreparedModelCache()
    request = _request()
    prepared = _prepared()
    calls = 0

    def prepare() -> PreparedAttentionCouplingModel:
        """Count the single expected miss preparation."""

        nonlocal calls
        calls += 1
        return prepared

    assert _resolve(cache, request, prepare) is prepared
    assert _resolve(cache, request, prepare) is prepared
    assert calls == 1


@pytest.mark.parametrize(
    "changed",
    [
        "model",
        "model_patches",
        "positive",
        "negative",
        "region_masks",
        "latent_image",
        "regional_prompt_weight",
        "region_mask_feather",
        "execution_mode",
    ],
)
def test_every_preparation_key_axis_causes_a_miss(changed: str) -> None:
    """Reject reuse whenever any preparation participant changes."""

    cache = AttentionCouplingPreparedModelCache()
    values = _request_values()
    original = _capture(values)
    _resolve(cache, original, _prepared)
    changed_values = values.copy()
    if changed == "model":
        model = values["model"]
        assert isinstance(model, SimpleNamespace)
        changed_values[changed] = SimpleNamespace(
            clone_base_uuid=model.clone_base_uuid,
            patches_uuid=model.patches_uuid,
        )
    elif changed == "model_patches":
        model = values["model"]
        assert isinstance(model, SimpleNamespace)
        changed_values["model"] = SimpleNamespace(
            clone_base_uuid=model.clone_base_uuid,
            patches_uuid=uuid4(),
        )
    elif changed == "positive":
        changed_values[changed] = [[torch.ones(1, 2, 3), {}]]
    elif changed == "negative":
        changed_values[changed] = [[torch.ones(1, 2, 3), {}]]
    elif changed == "region_masks":
        changed_values[changed] = torch.ones(1, 2, 2)
    elif changed == "latent_image":
        changed_values[changed] = {"samples": torch.ones(1, 4, 2, 2)}
    elif changed == "regional_prompt_weight":
        changed_values[changed] = 0.5
    elif changed == "region_mask_feather":
        changed_values[changed] = 1
    else:
        changed_values[changed] = RegionalAttentionExecutionMode.TILED
    replacement = _capture(changed_values)
    prepared = _prepared()

    assert _resolve(cache, replacement, lambda: prepared) is prepared
    assert cache.entry_count == 2


def test_nested_tensor_mutation_invalidates_request_identity() -> None:
    """Include tensor mutation versions without reading complete tensor values."""

    values = _request_values()
    original = _capture(values)
    positive = values["positive"]
    masks = values["region_masks"]
    assert isinstance(positive, list)
    context = positive[0][0]
    assert isinstance(context, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    context.add_(1.0)
    masks.mul_(0.5)

    assert _capture(values) != original


def test_conditioning_batch_tensor_mutation_invalidates_request_identity() -> None:
    """Traverse SimpleSyrup conditioning batches to their nested tensor versions."""

    values = _request_values()
    context = torch.zeros(1, 2, 3)
    values["positive"] = ConditioningBatch(([[context, {}]],))
    original = _capture(values)
    context.add_(1.0)

    assert _capture(values) != original


def test_inference_tensor_without_version_counter_is_snapshot_safe() -> None:
    """Retain exact inference-tensor identity when PyTorch omits mutation state."""

    values = _request_values()
    with torch.inference_mode():
        values["region_masks"] = torch.ones(1, 2, 2)

    assert isinstance(_capture(values), AttentionCouplingPreparedRequest)


def test_prepared_model_patch_mutation_invalidates_hit() -> None:
    """Never reuse a prepared patcher changed after cache publication."""

    cache = AttentionCouplingPreparedModelCache()
    request = _request()
    first = _prepared()
    _resolve(cache, request, lambda: first)
    first.model.patches_uuid = uuid4()
    replacement = _prepared()

    assert _resolve(cache, request, lambda: replacement) is replacement
    assert cache.entry_count == 1


def test_lru_eviction_refreshes_exact_hits() -> None:
    """Retain only the most recently used bounded request identities."""

    cache = AttentionCouplingPreparedModelCache(maximum_entries=2)
    requests = tuple(
        _request(regional_prompt_weight=value) for value in (0.1, 0.2, 0.3)
    )
    prepared = tuple(_prepared() for _ in requests)
    _resolve(cache, requests[0], lambda: prepared[0])
    _resolve(cache, requests[1], lambda: prepared[1])
    _resolve(cache, requests[0], lambda: _prepared())
    _resolve(cache, requests[2], lambda: prepared[2])
    replacement = _prepared()

    assert _resolve(cache, requests[1], lambda: replacement) is replacement
    assert cache.entry_count == 2


def test_concurrent_exact_resolution_prepares_once() -> None:
    """Serialize one exact miss while allowing every waiter to share its result."""

    cache = AttentionCouplingPreparedModelCache()
    request = _request()
    prepared = _prepared()
    callback_started = threading.Event()
    release_callback = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def prepare() -> PreparedAttentionCouplingModel:
        """Hold the first callback until all concurrent resolves are queued."""

        nonlocal calls
        with calls_lock:
            calls += 1
        callback_started.set()
        if not release_callback.wait(timeout=2.0):
            raise TimeoutError("Concurrent cache test did not release preparation.")
        return prepared

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = tuple(
            executor.submit(_resolve, cache, request, prepare) for _ in range(4)
        )
        assert callback_started.wait(timeout=2.0)
        release_callback.set()
        results = tuple(future.result(timeout=2.0) for future in futures)

    assert all(result is prepared for result in results)
    assert calls == 1


def test_failed_preparation_is_not_published() -> None:
    """Allow a later exact request to retry after one failed miss callback."""

    cache = AttentionCouplingPreparedModelCache()
    request = _request()

    def fail() -> PreparedAttentionCouplingModel:
        """Raise the expected preparation failure."""

        raise RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        _resolve(cache, request, fail)
    prepared = _prepared()

    assert cache.entry_count == 0
    assert _resolve(cache, request, lambda: prepared) is prepared


def test_disabled_policy_never_reuses_prepared_results() -> None:
    """Preserve the specialized Anima request lifecycle on every call."""

    cache = AttentionCouplingPreparedModelCache()
    request = _request()
    prepared = (_prepared(), _prepared())
    calls = 0

    def prepare() -> PreparedAttentionCouplingModel:
        """Return a distinct specialized result for every request."""

        nonlocal calls
        result = prepared[calls]
        calls += 1
        return result

    first = cache.resolve(
        request=request,
        policy=AttentionCouplingPreparedModelReuse.DISABLED,
        prepare=prepare,
    )
    second = cache.resolve(
        request=request,
        policy=AttentionCouplingPreparedModelReuse.DISABLED,
        prepare=prepare,
    )

    assert first is prepared[0]
    assert second is prepared[1]
    assert cache.entry_count == 0


def test_model_families_declare_architecture_owned_reuse_policy() -> None:
    """Keep exact reuse on standard UNet and Anima derivation unchanged."""

    assert (
        StandardUnetAttentionCouplingModelFamily().prepared_model_reuse
        is AttentionCouplingPreparedModelReuse.EXACT_REQUEST
    )
    assert (
        AnimaAttentionCouplingModelFamily().prepared_model_reuse
        is AttentionCouplingPreparedModelReuse.DISABLED
    )


def _resolve(
    cache: AttentionCouplingPreparedModelCache,
    request: AttentionCouplingPreparedRequest,
    prepare: Callable[[], PreparedAttentionCouplingModel],
) -> PreparedAttentionCouplingModel:
    """Resolve one exact-request cache entry through the standard policy."""

    return cache.resolve(
        request=request,
        policy=AttentionCouplingPreparedModelReuse.EXACT_REQUEST,
        prepare=prepare,
    )


def _request(
    *, regional_prompt_weight: float = 1.0
) -> AttentionCouplingPreparedRequest:
    """Capture one valid request with one selected scalar override."""

    values = _request_values()
    values["regional_prompt_weight"] = regional_prompt_weight
    return _capture(values)


def _capture(values: dict[str, object]) -> AttentionCouplingPreparedRequest:
    """Narrow one dynamic test mapping into the complete request constructor."""

    return AttentionCouplingPreparedRequest.capture(
        model=values["model"],
        positive=values["positive"],
        negative=values["negative"],
        region_masks=values["region_masks"],
        regional_prompt_weight=cast(float, values["regional_prompt_weight"]),
        region_mask_feather=cast(int, values["region_mask_feather"]),
        latent_image=values["latent_image"],
        execution_mode=cast(
            RegionalAttentionExecutionMode,
            values["execution_mode"],
        ),
    )


def _request_values() -> dict[str, object]:
    """Return one complete valid exact-request input mapping."""

    return {
        "model": SimpleNamespace(clone_base_uuid=uuid4(), patches_uuid=uuid4()),
        "positive": [[torch.zeros(1, 2, 3), {}]],
        "negative": [[torch.zeros(1, 2, 3), {}]],
        "region_masks": torch.zeros(1, 2, 2),
        "regional_prompt_weight": 1.0,
        "region_mask_feather": 0,
        "latent_image": {"samples": torch.zeros(1, 4, 2, 2)},
        "execution_mode": RegionalAttentionExecutionMode.FULL,
    }


def _prepared() -> PreparedAttentionCouplingModel:
    """Return one recognizable prepared result with mutable patch identity."""

    masks = RegionalMaskBank(
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2).clone(),
        canvas_width=2,
        canvas_height=2,
    )
    return PreparedAttentionCouplingModel(
        model=SimpleNamespace(patches_uuid=uuid4()),
        positive=object(),
        negative=object(),
        mask_bank=masks,
    )
