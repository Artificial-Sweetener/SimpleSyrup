# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode validated prompt-injected attention capture plans at runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionCaptureProfile,
    AttentionEvidenceMode,
    AttentionRegionControls,
    AttentionRegionRequest,
    AttentionRegionRequestKind,
)
from ..domain.attention_spatial_transform import (
    AttentionSpatialTransform,
    AttentionSpatialTransformKind,
)
from .comfy_graph_provenance import parse_graph_link


class AttentionCapturePlanCodec:
    """Own strict deserialization of internal prompt-plan JSON."""

    def decode(self, value: str) -> AttentionCapturePlan:
        """Return one validated domain plan from a serialized prompt input."""

        if not isinstance(value, str) or not value:
            raise ValueError("Attention capture plan JSON cannot be empty.")
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ValueError("Attention capture plan JSON must contain an object.")
        requests_value = payload.get("requests")
        if not isinstance(requests_value, list):
            raise ValueError("Attention capture plan requests must be a list.")
        requests = tuple(_request(item) for item in requests_value)
        return AttentionCapturePlan(
            sampler_node_id=_string(payload, "sampler_node_id"),
            model_owner_node_id=_string(payload, "model_owner_node_id"),
            model_input_name=_string(payload, "model_input_name"),
            model_link=_link(payload, "model_link"),
            positive_link=_link(payload, "positive_link"),
            requests=requests,
            prompt_text=_optional_string(payload, "prompt_text"),
            clip_link=_optional_link(payload, "clip_link"),
            source_aspect=_optional_float(payload, "source_aspect"),
        )


def _request(value: object) -> AttentionRegionRequest:
    """Decode one request and its validated controls."""

    if not isinstance(value, Mapping):
        raise ValueError("Attention capture request must be an object.")
    controls_value = value.get("controls")
    if not isinstance(controls_value, Mapping):
        raise ValueError("Attention capture request controls must be an object.")
    queries_value = value.get("queries")
    if not isinstance(queries_value, list) or any(
        not isinstance(query, str) for query in queries_value
    ):
        raise ValueError("Attention capture request queries must be strings.")
    controls = AttentionRegionControls(
        capture_start=_float(controls_value, "capture_start"),
        capture_end=_float(controls_value, "capture_end"),
        minimum_strength=_float(controls_value, "minimum_strength"),
        minimum_consensus=_float(controls_value, "minimum_consensus"),
        split_sensitivity=_float(controls_value, "split_sensitivity"),
        minimum_region_size=_integer(controls_value, "minimum_region_size"),
        instance_recall=_float(controls_value, "instance_recall"),
        geometry_recall=_float(controls_value, "geometry_recall"),
        keep_only=_integer(controls_value, "keep_only"),
        keep_by=_string(controls_value, "keep_by"),
        combine_segs=_boolean(controls_value, "combine_segs"),
        matte_solidity=_float(controls_value, "matte_solidity"),
        edge_feather=_integer(controls_value, "edge_feather"),
        profile=AttentionCaptureProfile(_string(controls_value, "profile")),
        evidence_mode=AttentionEvidenceMode(_string(controls_value, "evidence_mode")),
    )
    return AttentionRegionRequest(
        node_id=_string(value, "node_id"),
        kind=AttentionRegionRequestKind(_string(value, "kind")),
        queries=tuple(queries_value),
        controls=controls,
        sampler_stage=_integer(value, "sampler_stage"),
        spatial_transforms=_spatial_transforms(value.get("spatial_transforms")),
    )


def _string(mapping: Mapping[str, Any], key: str) -> str:
    """Return one required non-empty string field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Attention capture plan field '{key}' must be a string.")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    """Return one optional string field without coercion."""

    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Attention capture plan field '{key}' must be a string.")
    return value


def _float(mapping: Mapping[str, Any], key: str) -> float:
    """Return one required real field as float."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Attention capture plan field '{key}' must be numeric.")
    return float(value)


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    """Return one required integer field."""

    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"Attention capture plan field '{key}' must be an integer.")
    return value


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    """Return one required strict boolean field."""

    value = mapping.get(key)
    if type(value) is not bool:
        raise ValueError(f"Attention capture plan field '{key}' must be boolean.")
    return value


def _spatial_transforms(value: object) -> tuple[AttentionSpatialTransform, ...]:
    """Decode one request's ordered graph-visible spatial transforms."""

    if not isinstance(value, list):
        raise ValueError("Attention spatial transforms must be a list.")
    transforms: list[AttentionSpatialTransform] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Attention spatial transform must be an object.")
        raw_kind = _string(item, "kind")
        transforms.append(
            AttentionSpatialTransform(
                kind=AttentionSpatialTransformKind(raw_kind),
                width=_optional_integer(item, "width"),
                height=_optional_integer(item, "height"),
                scale=_optional_float(item, "scale"),
                anchor=str(item.get("anchor", "center")),
                divisible_by=_integer(item, "divisible_by"),
            )
        )
    return tuple(transforms)


def _optional_integer(mapping: Mapping[str, Any], key: str) -> int | None:
    """Return one optional strict integer field."""

    value = mapping.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"Attention capture plan field '{key}' must be an integer.")
    return value


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    """Return one optional real field."""

    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Attention capture plan field '{key}' must be numeric.")
    return float(value)


def _link(mapping: Mapping[str, Any], key: str) -> tuple[str, int]:
    """Return one required serialized graph link."""

    link = parse_graph_link(mapping.get(key))
    if link is None:
        raise ValueError(f"Attention capture plan field '{key}' must be a graph link.")
    return link


def _optional_link(
    mapping: Mapping[str, Any],
    key: str,
) -> tuple[str, int] | None:
    """Return one optional serialized graph link."""

    value = mapping.get(key)
    if value is None:
        return None
    link = parse_graph_link(value)
    if link is None:
        raise ValueError(f"Attention capture plan field '{key}' must be a graph link.")
    return link


ATTENTION_CAPTURE_PLAN_CODEC = AttentionCapturePlanCodec()
