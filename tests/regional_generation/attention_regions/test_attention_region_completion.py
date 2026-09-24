# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test observation-only selected-token attention capture."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import torch

from simple_syrup.domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionCaptureProfile,
    AttentionRegionControls,
    AttentionRegionRequest,
    AttentionRegionRequestKind,
)
from simple_syrup.domain.attention_region_maps import (
    AttentionTokenCatalog,
    AttentionTokenSpan,
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_region_affinity import (
    ATTENTION_AFFINITY_CALCULATOR,
)
from simple_syrup.runtime.attention_region_capture import AttentionRegionCaptureSession
from simple_syrup.runtime.attention_region_capture_backend import (
    OptimizedAttentionCaptureOverride,
)
from simple_syrup.runtime.attention_region_phrase_evidence import (
    specific_attention_head_weights,
)
from simple_syrup.runtime.attention_region_self_completion import (
    ATTENTION_REGION_SELF_COMPLETION,
    MAXIMUM_SELF_COMPLETION_ANCHORS,
    SpatialSelfAttention,
    _grid_anchor_indices,
)


def test_capture_pairs_spatial_self_attention_with_following_cross_call(
    monkeypatch: Any,
) -> None:
    """Pass same-layer self-attention only into derived concept evidence."""

    session = _session(profile=AttentionCaptureProfile.EXHAUSTIVE)
    observed_self_attention: list[SpatialSelfAttention | None] = []
    original = ATTENTION_AFFINITY_CALCULATOR.capture_spans

    def record_capture(*args: Any, **kwargs: Any) -> Any:
        """Record staged self-attention before delegating to affinity capture."""

        observed_self_attention.append(kwargs.get("self_attention"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ATTENTION_AFFINITY_CALCULATOR, "capture_spans", record_capture)
    spatial = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]]])
    session.observe(
        spatial,
        spatial,
        spatial,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )
    session.observe(
        spatial,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
        torch.ones(1, 3, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert len(observed_self_attention) == 1
    assert isinstance(observed_self_attention[0], SpatialSelfAttention)


def test_anima_capture_does_not_apply_sdxl_self_attention_completion(
    monkeypatch: Any,
) -> None:
    """Keep Anima concept capture on its validated cross-attention evidence path."""

    session = _session(
        profile=AttentionCaptureProfile.EXHAUSTIVE,
        sequence_length=512,
        model_family=RegionalModelFamily.ANIMA,
        source_aspect=0.75,
    )
    observed_self_attention: list[SpatialSelfAttention | None] = []
    original = ATTENTION_AFFINITY_CALCULATOR.capture_spans

    def record_capture(*args: Any, **kwargs: Any) -> Any:
        """Record completion input before delegating to affinity capture."""

        observed_self_attention.append(kwargs.get("self_attention"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ATTENTION_AFFINITY_CALCULATOR, "capture_spans", record_capture)
    spatial = torch.ones(1, 12, 2)
    session.observe(
        spatial,
        spatial,
        spatial,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )
    session.observe(
        spatial,
        torch.ones(1, 512, 2),
        torch.ones(1, 512, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert observed_self_attention == [None]


def test_anima_concept_evidence_preserves_contextualized_object_head() -> None:
    """Let phrase modifiers refine Anima object evidence without erasing its extent."""

    span = AttentionTokenSpan("pink hair", 1, (1, 2), (2,))
    session = _session(
        profile=AttentionCaptureProfile.EXHAUSTIVE,
        sequence_length=512,
        model_family=RegionalModelFamily.ANIMA,
        source_aspect=0.75,
        catalog_spans=(span,),
    )
    query = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.5, 1.5]]])
    key = torch.zeros(1, 512, 2)
    key[0, 1] = torch.tensor([1.0, 0.0])
    key[0, 2] = torch.tensor([0.0, 1.0])

    session.observe(
        query,
        key,
        torch.ones_like(key),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    attention_map = session.maps_for("search")[0]
    assert attention_map.concept_values is not None
    assert attention_map.concept_values[1] > attention_map.concept_values[0]
    assert attention_map.concept_values[1] > (attention_map.concept_values[2] * 0.75)


def test_anima_object_head_selection_rejects_diffuse_attention_heads() -> None:
    """Prefer a spatially specific object head over broad interaction context."""

    values = torch.tensor(
        [
            [
                [0.1, 0.1, 0.9, 0.8],
                [0.5, 0.5, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
                [0.3, 0.3, 0.3, 0.3],
            ]
        ]
    )

    weights = specific_attention_head_weights(values)

    assert weights.shape == (1, 4)
    assert weights[0, 0].item() == 1.0
    assert weights[0, 1:].count_nonzero().item() == 0


def test_self_completion_drops_weak_grid_anchors() -> None:
    """Keep spatial diversity without letting weak background cells steer completion."""

    seed = torch.arange(9, dtype=torch.float32).reshape(1, 9)

    anchors = _grid_anchor_indices(seed, 3, 3)

    assert anchors.shape == (1, MAXIMUM_SELF_COMPLETION_ANCHORS)
    assert set(anchors[0].tolist()) == {3, 4, 5, 6, 7, 8}


def test_self_completion_gates_related_recall_by_exact_object_grouping() -> None:
    """Admit a related strand while rejecting an equally strong unrelated region."""

    exact = torch.tensor([[1.0, 0.1, 0.05, 0.05]])
    related = torch.tensor([[1.0, 0.1, 0.9, 0.9]])
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]])
    self_attention = SpatialSelfAttention(query=query, key=query)

    completed = ATTENTION_REGION_SELF_COMPLETION.complete(
        exact,
        self_attention,
        2,
        2,
        related_seed=related,
    )

    assert completed[0, 2] > completed[0, 3]


def test_sibling_consumers_share_one_concurrent_materialization(
    monkeypatch: Any,
) -> None:
    """Prevent parallel downstream nodes from consuming an emptied capture."""

    session = _session(profile=AttentionCaptureProfile.EXHAUSTIVE)
    session.observe(
        torch.ones(1, 2, 2),
        torch.ones(1, 3, 2),
        torch.ones(1, 3, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )
    original = ATTENTION_AFFINITY_CALCULATOR.materialize
    started = Event()
    release = Event()
    call_count = 0

    def delayed_materialize(*args: Any, **kwargs: Any) -> Any:
        """Hold the first materializer until a sibling is waiting."""

        nonlocal call_count
        call_count += 1
        started.set()
        assert release.wait(timeout=2.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ATTENTION_AFFINITY_CALCULATOR,
        "materialize",
        delayed_materialize,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(session.maps_for, "search")
        assert started.wait(timeout=2.0)
        second = executor.submit(session.maps_for, "search")
        release.set()
        results = (first.result(timeout=2.0), second.result(timeout=2.0))

    assert call_count == 1
    assert all(len(result) == 1 for result in results)
    assert results[0] is not results[1]
    assert results[0][0].values.equal(results[1][0].values)


def test_capture_profile_subsamples_calls_without_changing_attention_output() -> None:
    """Delegate every denoising call while retaining only fast-profile samples."""

    session = _session(profile=AttentionCaptureProfile.FAST)
    override = OptimizedAttentionCaptureOverride(session, None)
    query = torch.ones(1, 2, 2)
    key = torch.ones(1, 3, 2)
    value = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    def original(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a sentinel output while accepting Comfy attention arguments."""

        del args, kwargs
        return torch.full((1, 2, 2), 7.0)

    options = tuple(
        _options(cond_or_uncond=[0], block_index=index) for index in range(33)
    )
    outputs = tuple(
        override(
            original,
            query,
            key,
            value,
            1,
            transformer_options=options[index],
        )
        for index in range(33)
    )

    assert all(torch.equal(output, outputs[0]) for output in outputs)
    assert outputs[0].eq(7.0).all().item()
    assert len(session.maps_for("search")) == 2


def test_fast_capture_rotates_sampled_layers_between_denoising_steps() -> None:
    """Cover a different layer offset at each step without increasing stride cost."""

    session = _session(profile=AttentionCaptureProfile.FAST)
    query = torch.ones(1, 2, 2)
    key = torch.ones(1, 3, 2)
    value = torch.ones(1, 3, 2)
    for block_index in range(33):
        session.observe(
            query,
            key,
            value,
            1,
            _options(cond_or_uncond=[0], sigma=1.0, block_index=block_index),
            skip_reshape=False,
        )
    for block_index in range(2):
        session.observe(
            query,
            key,
            value,
            1,
            _options(cond_or_uncond=[0], sigma=0.5, block_index=block_index),
            skip_reshape=False,
        )

    assert len(session.maps_for("search")) == 3


def test_fast_capture_subsamples_spatial_self_completion(
    monkeypatch: Any,
) -> None:
    """Retain sparse self-attention recall without paying for every fast sample."""

    session = _session(profile=AttentionCaptureProfile.FAST)
    observed: list[SpatialSelfAttention | None] = []
    original = ATTENTION_AFFINITY_CALCULATOR.capture_spans

    def capture_spans(*args: Any, **kwargs: Any) -> Any:
        """Record completion inputs while preserving affinity behavior."""

        observed.append(kwargs.get("self_attention"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ATTENTION_AFFINITY_CALCULATOR,
        "capture_spans",
        capture_spans,
    )
    for block_index in range(65):
        options = _options(cond_or_uncond=[0], block_index=block_index)
        session.observe(
            torch.ones(1, 2, 2),
            torch.ones(1, 2, 2),
            torch.ones(1, 2, 2),
            1,
            options,
            skip_reshape=False,
        )
        session.observe(
            torch.ones(1, 2, 2),
            torch.ones(1, 3, 2),
            torch.ones(1, 3, 2),
            1,
            options,
            skip_reshape=False,
        )

    assert len(observed) == 3
    assert sum(isinstance(value, SpatialSelfAttention) for value in observed) == 1


def test_coalesced_requests_share_one_unioned_native_affinity_pass(
    monkeypatch: Any,
) -> None:
    """Calculate shared prompt-token affinities once and route maps per request."""

    controls = AttentionRegionControls(
        0.0,
        1.0,
        0.3,
        0.2,
        0.5,
        1,
        AttentionCaptureProfile.EXHAUSTIVE,
    )
    requests = (
        AttentionRegionRequest(
            "all",
            AttentionRegionRequestKind.ALL_PROMPT_SEGS,
            (),
            controls,
        ),
        AttentionRegionRequest(
            "hair",
            AttentionRegionRequestKind.CONCEPT_SEGS,
            ("pink hair",),
            controls,
        ),
    )
    plan = AttentionCapturePlan(
        "sampler",
        "sampler",
        "model",
        ("model", 0),
        ("positive", 0),
        requests,
        "1girl, pink hair",
        ("loader", 1),
    )
    spans = (
        AttentionTokenSpan("1girl", 1, (1,)),
        AttentionTokenSpan("pink hair", 1, (2,)),
    )
    session = AttentionRegionCaptureSession(
        plan=plan,
        model_family=RegionalModelFamily.STANDARD_UNET,
        token_catalog=AttentionTokenCatalog(4, spans, (0, 1, 2, 3)),
        request_spans={"hair": (spans[1],), "all": spans},
    )
    original = ATTENTION_AFFINITY_CALCULATOR.capture_spans
    calls: list[tuple[AttentionTokenSpan, ...]] = []

    def record_capture(*args: Any, **kwargs: Any) -> Any:
        """Record the unioned spans before delegating to real affinity math."""

        captured_spans = kwargs.get("spans", args[3])
        calls.append(captured_spans)
        return original(*args, **kwargs)

    monkeypatch.setattr(ATTENTION_AFFINITY_CALCULATOR, "capture_spans", record_capture)
    session.observe(
        torch.ones(1, 2, 2),
        torch.ones(1, 4, 2),
        torch.ones(1, 4, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert calls == [spans]
    assert tuple(value.label for value in session.maps_for("hair")) == ("pink hair",)
    assert tuple(value.label for value in session.maps_for("all")) == (
        "1girl",
        "pink hair",
    )


def _session(
    profile: AttentionCaptureProfile = AttentionCaptureProfile.EXHAUSTIVE,
    sequence_length: int = 3,
    source_aspect: float | None = None,
    model_family: RegionalModelFamily = RegionalModelFamily.STANDARD_UNET,
    token_indices: tuple[int, ...] = (1,),
    catalog_spans: tuple[AttentionTokenSpan, ...] | None = None,
) -> AttentionRegionCaptureSession:
    """Return one exact-native-query capture session."""

    controls = AttentionRegionControls(0.0, 1.0, 0.3, 0.2, 0.5, 1, profile)
    request = AttentionRegionRequest(
        "search",
        AttentionRegionRequestKind.CONCEPT_SEGS,
        ("pink hair",),
        controls,
    )
    plan = AttentionCapturePlan(
        "sampler",
        "sampler",
        "model",
        ("model", 0),
        ("positive", 0),
        (request,),
        "pink hair",
        ("loader", 1),
        source_aspect,
    )
    default_span = AttentionTokenSpan("pink hair", 1, token_indices)
    catalog = AttentionTokenCatalog(
        sequence_length,
        catalog_spans or (default_span,),
        tuple(range(sequence_length)),
    )
    return AttentionRegionCaptureSession(
        plan=plan,
        model_family=model_family,
        token_catalog=catalog,
        request_spans={"search": (catalog.spans[0],)},
    )


def _options(
    *,
    cond_or_uncond: list[int],
    sigma: float = 1.0,
    block_index: int = 0,
) -> dict[str, object]:
    """Return exact sampler metadata at the beginning of denoising."""

    return {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "sigmas": torch.tensor([sigma]),
        "cond_or_uncond": cond_or_uncond,
        "block": ("middle", 0),
        "block_index": block_index,
    }


def _patcher() -> Any:
    """Create a real CPU ModelPatcher with isolated transformer options."""

    from comfy.model_patcher import ModelPatcher

    base_model = torch.nn.Module()
    base_model.diffusion_model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    return ModelPatcher(base_model, load_device=device, offload_device=device)
