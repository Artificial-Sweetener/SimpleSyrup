# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test observation-only selected-token attention capture."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
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
    OpenVocabularyContext,
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_region_affinity import (
    ATTENTION_AFFINITY_CALCULATOR,
    _select_relevant_head_maps,
)
from simple_syrup.runtime.attention_region_capture import AttentionRegionCaptureSession
from simple_syrup.runtime.attention_region_capture_backend import (
    ATTENTION_REGION_CAPTURE_BACKEND,
    OptimizedAttentionCaptureOverride,
)
from simple_syrup.runtime.attention_region_contextual_spans import (
    ATTENTION_CONTEXTUAL_SPAN_SELECTOR,
)
from simple_syrup.runtime.attention_region_open_vocabulary import (
    OpenVocabularyKeyProjector,
)
from simple_syrup.runtime.attention_region_phrase_evidence import (
    ANIMA_PHRASE_EVIDENCE_SERVICE,
    specific_attention_head_weights,
)
from simple_syrup.runtime.attention_region_self_completion import (
    ATTENTION_REGION_SELF_COMPLETION,
    MAXIMUM_SELF_COMPLETION_ANCHORS,
    SpatialSelfAttention,
    _grid_anchor_indices,
)


def test_attention_controls_reject_invalid_instance_recall() -> None:
    """Reject disconnected-instance recall outside its normalized range."""

    with pytest.raises(ValueError, match="instance recall"):
        AttentionRegionControls(
            0.0,
            1.0,
            0.15,
            0.25,
            0.0,
            1,
            AttentionCaptureProfile.FAST,
            instance_recall=1.01,
        )


def test_attention_controls_reject_invalid_geometry_recall() -> None:
    """Reject connected-geometry recall outside its normalized range."""

    with pytest.raises(ValueError, match="geometry recall"):
        AttentionRegionControls(
            0.0,
            1.0,
            0.15,
            0.25,
            0.0,
            1,
            AttentionCaptureProfile.FAST,
            geometry_recall=-0.01,
        )


def test_capture_selects_positive_rows_and_exact_prompt_tokens() -> None:
    """Capture one native concept without retaining a full attention matrix."""

    session = _session(profile=AttentionCaptureProfile.EXHAUSTIVE)
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    key = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]],
        ]
    )

    session.observe(
        query,
        key,
        key,
        1,
        _options(cond_or_uncond=[0, 1]),
        skip_reshape=False,
    )

    maps = session.maps_for("search")
    assert len(maps) == 1
    assert maps[0].label == "pink hair"
    assert maps[0].values.device.type == "cpu"
    assert maps[0].values[0].item() > maps[0].values[1].item()


def test_capture_retains_value_aware_phrase_evidence_beside_raw_attention() -> None:
    """Favor the phrase token carrying stronger projected model contribution."""

    session = _session(
        profile=AttentionCaptureProfile.EXHAUSTIVE,
        sequence_length=4,
        token_indices=(1, 2),
    )
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])
    value = torch.tensor([[[1.0, 1.0], [0.1, 0.1], [4.0, 4.0], [1.0, 1.0]]])

    session.observe(
        query,
        key,
        value,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    attention_map = session.maps_for("search")[0]
    assert torch.isclose(attention_map.values[0], attention_map.values[1])
    assert attention_map.concept_values is not None
    assert attention_map.concept_values[1] > attention_map.concept_values[0]
    assert attention_map.uniform_probability == 0.25


def test_contextual_span_selector_admits_only_distinct_related_prompt_spans() -> None:
    """Select a related conditioning span while rejecting an orthogonal concept."""

    target = AttentionTokenSpan("pink hair", 1, (1,))
    related = AttentionTokenSpan("twintails", 1, (2,))
    unrelated = AttentionTokenSpan("smile", 1, (3,))
    key = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]]])

    selections = ATTENTION_CONTEXTUAL_SPAN_SELECTOR.select(
        key=key,
        targets=(target,),
        candidates=(target, related, unrelated),
    )

    assert selections[target].token_indices == (1, 2)
    assert selections[target].token_weights[0] == 1.0
    assert 0.0 < selections[target].token_weights[1] < 1.0


def test_contextual_span_selector_uses_object_head_instead_of_modifier_color() -> None:
    """Relate an object phrase through its head without following a color token."""

    target = AttentionTokenSpan("pink outfit", 1, (1, 2), (2,))
    related_part = AttentionTokenSpan("short skirt", 1, (3, 4), (4,))
    color_only = AttentionTokenSpan("pink petals", 1, (5,), (5,))
    key = torch.tensor(
        [
            [
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    selection = ATTENTION_CONTEXTUAL_SPAN_SELECTOR.select(
        key=key,
        targets=(target,),
        candidates=(target, related_part, color_only),
    )[target]

    assert selection.token_indices == (1, 2, 4)
    assert selection.token_weights[0] < selection.token_weights[1]


def test_anima_compound_phrase_uses_specific_modifier_to_constrain_its_head() -> None:
    """Keep a small compound concept local when its noun head is spatially broad."""

    probability = torch.tensor(
        [
            [
                [
                    [0.8, 1.0, 1.0],
                    [0.8, 0.9, 0.9],
                    [0.8, 0.05, 0.7],
                    [0.8, 0.05, 0.7],
                ]
            ]
        ]
    )
    span = AttentionTokenSpan(
        "blue butterfly ornaments",
        1,
        (0, 1, 2),
        (2,),
    )

    concept = ANIMA_PHRASE_EVIDENCE_SERVICE.derive(
        probability=probability,
        span=span,
        union_positions={0: 0, 1: 1, 2: 2},
    )

    assert concept[0, 0] > 0.8
    assert concept[0, 1] > 0.7
    assert concept[0, 2] < concept[0, 0] * 0.5
    assert concept[0, 3] < concept[0, 0] * 0.5


def test_anima_broad_modifier_does_not_erase_extended_head_geometry() -> None:
    """Preserve a noun silhouette when its only modifier carries no locality."""

    probability = torch.tensor(
        [
            [
                [
                    [0.6, 1.0],
                    [0.6, 0.8],
                    [0.6, 0.5],
                    [0.6, 0.3],
                ]
            ]
        ]
    )
    span = AttentionTokenSpan("pink hair", 1, (0, 1), (1,))

    concept = ANIMA_PHRASE_EVIDENCE_SERVICE.derive(
        probability=probability,
        span=span,
        union_positions={0: 0, 1: 1},
    )

    assert torch.allclose(concept[0], probability[0, 0, :, 1].to(torch.float16))


def test_anima_single_token_concept_uses_noun_head_evidence_directly() -> None:
    """Handle noun-only prompt segments without requiring modifier positions."""

    probability = torch.tensor([[[[1.0], [0.8], [0.3], [0.0]]]])
    span = AttentionTokenSpan("twintails", 1, (0,), (0,))

    concept = ANIMA_PHRASE_EVIDENCE_SERVICE.derive(
        probability=probability,
        span=span,
        union_positions={0: 0},
    )

    assert torch.equal(concept[0], probability[0, 0, :, 0].to(torch.float16))


def test_anima_related_prompt_head_recovers_a_disjoint_concept_part() -> None:
    """Admit a related prompt noun at reduced strength without replacing the core."""

    probability = torch.tensor(
        [
            [
                [
                    [0.5, 1.0, 0.0],
                    [0.5, 0.2, 0.0],
                    [0.5, 0.0, 0.9],
                    [0.5, 0.0, 0.8],
                ]
            ]
        ]
    )
    span = AttentionTokenSpan("pink hair", 1, (0, 1), (1,))

    concept = ANIMA_PHRASE_EVIDENCE_SERVICE.derive(
        probability=probability,
        span=span,
        union_positions={0: 0, 1: 1, 2: 2},
        contextual_token_indices=(0, 1, 2),
        contextual_token_weights=(0.45, 1.0, 0.3),
    )

    assert concept[0, 0] == 1.0
    assert concept[0, 2] > 0.25
    assert concept[0, 3] > 0.2


def test_concept_head_selection_rejects_a_spatially_disagreeing_head() -> None:
    """Prefer concept heads that agree on the object while retaining their detail."""

    probability = torch.tensor(
        [
            [
                [[0.9], [0.8], [0.4], [0.0]],
                [[0.8], [0.9], [0.0], [0.0]],
                [[0.7], [0.8], [0.0], [0.0]],
                [[0.0], [0.0], [0.9], [0.9]],
            ]
        ]
    )

    _weighted, selected = _select_relevant_head_maps(
        probability,
        torch.ones(1, 4, 1),
    )

    assert selected[0, 0, 0] > selected[0, 3, 0]
    assert selected[0, 1, 0] > selected[0, 3, 0]


def test_related_tokens_use_heads_selected_by_the_exact_concept() -> None:
    """Recover related detail without admitting its independently face-focused head."""

    probability = torch.tensor(
        [
            [
                [[0.9, 0.1], [0.8, 0.1], [0.4, 0.8], [0.0, 0.0]],
                [[0.8, 0.1], [0.9, 0.1], [0.4, 0.7], [0.0, 0.0]],
                [[0.7, 0.1], [0.8, 0.1], [0.3, 0.6], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.1], [0.9, 0.9]],
            ]
        ]
    )

    _weighted, selected = _select_relevant_head_maps(
        probability,
        torch.ones(1, 4, 2),
        exact_token_mask=torch.tensor([True, False]),
    )

    assert selected[0, 2, 1] > selected[0, 3, 1]


def test_concept_capture_enriches_related_span_without_changing_raw_attention() -> None:
    """Recover self-grouped related evidence only in the derived concept channel."""

    target = AttentionTokenSpan("pink hair", 1, (1,))
    related = AttentionTokenSpan("twintails", 1, (2,))
    unrelated = AttentionTokenSpan("smile", 1, (3,))
    session = _session(
        profile=AttentionCaptureProfile.EXHAUSTIVE,
        sequence_length=4,
        catalog_spans=(target, related, unrelated),
    )
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.0, -1.0]]])
    value = torch.ones_like(key)

    spatial = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    session.observe(
        spatial,
        spatial,
        spatial,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )
    session.observe(
        query,
        key,
        value,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    attention_map = session.maps_for("search")[0]
    assert attention_map.concept_values is not None
    raw_ratio = attention_map.values[1] / attention_map.values[0]
    concept_ratio = attention_map.concept_values[1] / attention_map.concept_values[0]
    assert concept_ratio > raw_ratio


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


def test_backend_clones_model_and_composes_existing_attention_override() -> None:
    """Keep source options intact while preserving an upstream override."""

    source = _patcher()

    def previous(
        original: object,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return a sentinel from an admitted upstream override."""

        del original, args, kwargs
        return torch.tensor(3.0)

    source.model_options["transformer_options"]["optimized_attention_override"] = (
        previous
    )

    derived: Any = ATTENTION_REGION_CAPTURE_BACKEND.derive(source, _session())

    assert derived.parent is source
    assert (
        source.model_options["transformer_options"]["optimized_attention_override"]
        is previous
    )
    installed = derived.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert isinstance(installed, OptimizedAttentionCaptureOverride)
    assert (
        installed(
            lambda *_args, **_kwargs: torch.tensor(1.0),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            1,
        ).item()
        == 3.0
    )


def test_open_vocabulary_query_projects_side_keys_without_changing_native_keys() -> (
    None
):
    """Reuse SDXL spatial queries for an absent phrase without conditioning edits."""

    controls = AttentionRegionControls(
        0.0,
        1.0,
        0.3,
        0.2,
        0.5,
        1,
        AttentionCaptureProfile.EXHAUSTIVE,
    )
    request = AttentionRegionRequest(
        "search",
        AttentionRegionRequestKind.CONCEPT_SEGS,
        ("head",),
        controls,
    )
    plan = AttentionCapturePlan(
        "sampler",
        "sampler",
        "model",
        ("model", 0),
        ("positive", 0),
        (request,),
        "1girl",
        ("loader", 1),
    )
    context = OpenVocabularyContext(
        "head",
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]),
        (1,),
    )
    session = AttentionRegionCaptureSession(
        plan=plan,
        model_family=RegionalModelFamily.STANDARD_UNET,
        token_catalog=AttentionTokenCatalog(
            3, (AttentionTokenSpan("1girl", 1, (1,)),), (1, 2, 3)
        ),
        request_spans={"search": ()},
        open_vocabulary_contexts=(context,),
    )
    projector = OpenVocabularyKeyProjector(torch.nn.Identity(), (context,), session)
    native_context = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])

    native_keys = projector(native_context)
    session.observe(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        native_keys,
        native_context,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert torch.equal(native_keys, native_context)
    maps = session.maps_for("search")
    assert len(maps) == 1
    assert maps[0].label == "head"
    assert maps[0].values[0].item() > maps[0].values[1].item()


def test_open_vocabulary_query_projection_is_cached_per_device_and_dtype() -> None:
    """Avoid repeating absent-query key projections at every denoising call."""

    class CountingProjection(torch.nn.Module):
        """Count native and side-projection calls while preserving values."""

        calls: int

        def __init__(self) -> None:
            """Initialize an unused projection counter."""

            super().__init__()
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Return the input after recording the projection."""

            self.calls += 1
            return value

    context = OpenVocabularyContext(
        "head",
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]),
        (1,),
    )
    projection = CountingProjection()
    projector = OpenVocabularyKeyProjector(projection, (context,), _session())
    native_context = torch.ones(1, 3, 2)

    assert torch.equal(projector(native_context), native_context)
    assert torch.equal(projector(native_context), native_context)
    assert projection.calls == 3


def test_sampled_denominator_bounds_unusually_strong_selected_keys() -> None:
    """Keep fast-profile maps finite when denominator sampling misses the peak key."""

    session = _session(
        profile=AttentionCaptureProfile.FAST,
        sequence_length=40,
    )
    query = torch.tensor([[[1000.0, 0.0], [0.0, 1.0]]])
    key = torch.zeros(1, 40, 2)
    key[0, 1, 0] = 1000.0

    session.observe(
        query,
        key,
        key,
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    maps = session.maps_for("search")
    assert len(maps) == 1
    assert torch.isfinite(maps[0].values).all().item()
    assert maps[0].values.max().item() <= 1.0


def test_graph_source_aspect_orients_dit_geometry_without_transformer_metadata() -> (
    None
):
    """Use the selected sampler's portrait input when a DiT omits shape metadata."""

    session = _session(source_aspect=0.75)

    session.observe(
        torch.ones(1, 12, 2),
        torch.ones(1, 3, 2),
        torch.ones(1, 3, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    attention_map = session.maps_for("search")[0]
    assert (attention_map.spatial_height, attention_map.spatial_width) == (4, 3)


def test_anima_without_graph_or_runtime_geometry_fails_closed() -> None:
    """Return no maps instead of guessing an unprojectable Anima grid orientation."""

    session = _session(
        sequence_length=512,
        model_family=RegionalModelFamily.ANIMA,
    )

    session.observe(
        torch.ones(1, 12, 2),
        torch.ones(1, 512, 2),
        torch.ones(1, 512, 2),
        1,
        _options(cond_or_uncond=[0]),
        skip_reshape=False,
    )

    assert session.maps_for("search") == ()
    assert "not graph-visible" in session.status_message


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
