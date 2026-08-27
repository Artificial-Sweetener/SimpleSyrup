# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test public attention-region v3 contracts and shared execution shapes."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from simple_syrup.domain.attention_region_capture import (
    AttentionCaptureProfile,
    AttentionRegionControls,
)
from simple_syrup.domain.attention_region_maps import CapturedAttentionMap
from simple_syrup.nodes_v3 import get_nodes
from simple_syrup.nodes_v3.attention_capture_model import AttentionCaptureModelV3
from simple_syrup.runtime.attention_region_store import ATTENTION_REGION_CAPTURE_STORE
from simple_syrup.services.attention_capture_model_service import (
    EmptyAttentionCaptureSession,
)
from simple_syrup.services.attention_region_node_service import (
    ATTENTION_REGION_NODE_SERVICE,
)


def test_v3_registry_exposes_four_public_nodes_and_internal_capture_node() -> None:
    """Pin public identifiers and keep the injected implementation node dev-only."""

    schemas = {
        cast(Any, node).define_schema().node_id: cast(Any, node).define_schema()
        for node in get_nodes()
    }

    assert {
        "SimpleSyrup.ConceptAttentionSEGS",
        "SimpleSyrup.AllPromptAttentionSEGS",
        "SimpleSyrup.AttentionRegionMask",
        "SimpleSyrup.AttentionMaskedConditioning",
        "SimpleSyrup.AttentionCaptureModel",
    } <= schemas.keys()
    assert schemas["SimpleSyrup.AttentionCaptureModel"].is_dev_only is True
    for node_id in (
        "SimpleSyrup.ConceptAttentionSEGS",
        "SimpleSyrup.AllPromptAttentionSEGS",
        "SimpleSyrup.AttentionRegionMask",
        "SimpleSyrup.AttentionMaskedConditioning",
    ):
        schema = schemas[node_id]
        assert schema.description
        assert all(item.tooltip for item in (*schema.inputs, *schema.outputs))

    concept_inputs = tuple(
        item.id for item in schemas["SimpleSyrup.ConceptAttentionSEGS"].inputs
    )
    assert "concepts" in concept_inputs
    assert "queries" not in concept_inputs
    assert {
        "sampler_stage",
        "evidence_mode",
        "keep_only",
        "keep_by",
        "combine_segs",
        "matte_solidity",
        "edge_feather",
    } <= set(concept_inputs)
    concept_defaults = {
        item.id: item.default
        for item in schemas["SimpleSyrup.ConceptAttentionSEGS"].inputs
        if hasattr(item, "default")
    }
    assert concept_defaults["minimum_strength"] == 0.15
    assert concept_defaults["evidence_mode"] == "concept isolation"
    assert concept_defaults["split_sensitivity"] == 0.0
    assert concept_defaults["minimum_region_size"] == 512
    assert concept_defaults["keep_only"] == 1
    assert concept_defaults["matte_solidity"] == 0.75
    all_prompt_defaults = {
        item.id: item.default
        for item in schemas["SimpleSyrup.AllPromptAttentionSEGS"].inputs
        if hasattr(item, "default")
    }
    assert all_prompt_defaults["evidence_mode"] == "raw attention"


def test_internal_capture_model_is_never_reused_across_prompt_executions() -> None:
    """Republish ephemeral capture state even when graph inputs are unchanged."""

    assert math.isnan(AttentionCaptureModelV3.fingerprint_inputs())


def test_image_service_preserves_pixels_and_returns_batch_segs_and_soft_masks() -> None:
    """Keep BHWC pixels bit-identical while rendering one SEGS payload per image."""

    image = torch.rand(2, 4, 4, 3)
    session = _Session(
        (
            CapturedAttentionMap(
                "hair",
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
                0.5,
                "layer",
                0,
            ),
            CapturedAttentionMap(
                "hair",
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
                0.5,
                "layer",
                1,
            ),
        )
    )
    ATTENTION_REGION_CAPTURE_STORE.publish(("search",), session)

    result = ATTENTION_REGION_NODE_SERVICE.for_image(
        request_node_id="search",
        image=image,
        controls=_controls(),
    )

    assert result.image.data_ptr() == image.data_ptr()
    assert torch.equal(result.image, image)
    assert len(result.segs) == 2
    assert result.mask.shape == (2, 4, 4)
    assert result.mask[0].argmax().item() != result.mask[1].argmax().item()


def test_latent_service_supports_anima_singleton_frame_and_empty_no_op() -> None:
    """Return exact BCTHW provenance and correctly sized empty masks."""

    latent: dict[str, object] = {"samples": torch.zeros(2, 16, 1, 8, 6)}
    ATTENTION_REGION_CAPTURE_STORE.publish(
        ("mask",),
        EmptyAttentionCaptureSession(("mask",), "unsupported model"),
    )

    result = ATTENTION_REGION_NODE_SERVICE.for_latent(
        request_node_id="mask",
        latent=latent,
        controls=_controls(),
    )

    assert result.latent is latent
    assert result.mask.shape == (2, 8, 6)
    assert result.mask.count_nonzero().item() == 0


def test_mask_conditioning_uses_comfy_generic_mask_metadata(monkeypatch: Any) -> None:
    """Attach a later-pass mask through Comfy's conditioning helper contract."""

    calls: list[tuple[object, dict[str, object]]] = []

    def setter(conditioning: object, values: dict[str, object]) -> object:
        """Record exact conditioning metadata and return a sentinel."""

        calls.append((conditioning, values))
        return "masked"

    fake_module = type(
        "NodeHelpers",
        (),
        {"conditioning_set_values": staticmethod(setter)},
    )()
    import simple_syrup.services.attention_region_node_service as service_module

    monkeypatch.setattr(service_module, "import_module", lambda _name: fake_module)
    mask = torch.ones(1, 4, 4)

    result = ATTENTION_REGION_NODE_SERVICE.mask_conditioning("cond", mask, 1.25)

    assert result == "masked"
    assert calls == [
        (
            "cond",
            {
                "mask": mask,
                "set_area_to_bounds": False,
                "mask_strength": 1.25,
            },
        )
    ]


class _Session:
    """Expose deterministic maps through the capture-store protocol."""

    status_message = "captured"

    def __init__(self, maps: tuple[CapturedAttentionMap, ...]) -> None:
        """Retain immutable maps for one request."""

        self._maps = maps

    def maps_for(self, request_node_id: str) -> tuple[CapturedAttentionMap, ...]:
        """Return maps for the test request."""

        assert request_node_id == "search"
        return self._maps


def _controls() -> AttentionRegionControls:
    """Return permissive deterministic rendering controls."""

    return AttentionRegionControls(
        0.0,
        1.0,
        0.1,
        0.0,
        0.0,
        1,
        AttentionCaptureProfile.EXHAUSTIVE,
    )
