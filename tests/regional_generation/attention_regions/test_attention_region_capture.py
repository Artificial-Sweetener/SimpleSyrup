# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test observation-only selected-token attention capture."""

from __future__ import annotations

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
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_region_affinity import (
    _select_relevant_head_maps,
)
from simple_syrup.runtime.attention_region_capture import AttentionRegionCaptureSession
from simple_syrup.runtime.attention_region_contextual_spans import (
    ATTENTION_CONTEXTUAL_SPAN_SELECTOR,
)
from simple_syrup.runtime.attention_region_phrase_evidence import (
    ANIMA_PHRASE_EVIDENCE_SERVICE,
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
