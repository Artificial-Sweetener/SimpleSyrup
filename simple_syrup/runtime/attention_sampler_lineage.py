# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trace ordered sampler lineage through known spatial graph boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..domain.attention_sampler_lineage import (
    AttentionSamplerLineage,
    AttentionSamplerStage,
)
from ..domain.attention_spatial_transform import (
    AttentionSpatialTransform,
    AttentionSpatialTransformKind,
)
from ..domain.graph_provenance import BrokenProvenance, GraphLink
from .comfy_graph_provenance import MAX_PROVENANCE_HOPS, NodeRegistry, parse_graph_link

_DIRECT_SAMPLERS = frozenset({"KSampler", "KSamplerAdvanced", "SamplerCustom"})
_GUIDER_SAMPLER = "SamplerCustomAdvanced"
_FULL_CANVAS_DETAILERS = frozenset({"SimpleSyrup.DetailSEGSAsRegions"})
_CROP_LOCAL_DETAILERS = frozenset(
    {
        "SimpleSyrup.DetailSEGSByScaleFactor",
        "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion",
        "DetailerForEach",
    }
)
_IMAGE_TO_LATENT = {
    "VAEDecode": "samples",
    "SimpleSyrup.VAEDecodeOptions": "samples",
}
_LATENT_TO_IMAGE = {
    "VAEEncode": "pixels",
    "VAEEncodeForInpaint": "pixels",
    "SimpleSyrup.VAEEncodeOptions": "pixels",
    "SimpleSyrup.SimpleVAEEncode": "image",
}
_IMAGE_TRANSFORMS = {
    "ImageScale": "image",
    "ImageScaleBy": "image",
    "ImageUpscaleWithModel": "image",
    "SimpleSyrup.ResizeImageToTarget": "image",
}
_LATENT_TRANSFORMS = {
    "LatentUpscale": "samples",
    "LatentUpscaleBy": "samples",
}


@dataclass(frozen=True, slots=True)
class _Cursor:
    """Track one graph link and its spatial value kind."""

    link: GraphLink
    kind: str


class AttentionSamplerLineageResolver:
    """Discover every recognized sampler on one connected spatial ancestry."""

    def resolve(
        self,
        *,
        prompt: Mapping[str, Any],
        start_link: GraphLink,
        source_kind: str,
        node_registry: NodeRegistry,
    ) -> AttentionSamplerLineage | BrokenProvenance:
        """Return chronological stages or an exact provenance failure."""

        current = _Cursor(start_link, source_kind)
        visited: set[GraphLink] = set()
        reverse_stages: list[AttentionSamplerStage] = []
        reverse_transforms: list[AttentionSpatialTransform] = []
        for _hop in range(MAX_PROVENANCE_HOPS):
            if current.link in visited:
                return BrokenProvenance(
                    "sampler lineage contains a cycle", node_id=current.link[0]
                )
            visited.add(current.link)
            node = _node(prompt, current.link[0])
            if node is None:
                return BrokenProvenance(
                    "sampler lineage source node is missing", node_id=current.link[0]
                )
            class_type = _class_type(node)
            inputs = _inputs(node)
            stage = _sampling_stage(
                prompt,
                current,
                class_type,
                inputs,
                node_registry,
            )
            if isinstance(stage, BrokenProvenance):
                return stage
            if stage is not None:
                stage = replace(
                    stage,
                    forward_transforms=tuple(reversed(reverse_transforms)),
                )
                reverse_stages.append(stage)
                current = _Cursor(stage.upstream_link, stage.upstream_kind)
                continue
            next_cursor = _spatial_predecessor(
                current, class_type, inputs, node_registry
            )
            if isinstance(next_cursor, BrokenProvenance):
                if reverse_stages and _is_spatial_terminal(inputs):
                    break
                return next_cursor
            transform = _spatial_transform(class_type, inputs)
            if isinstance(transform, BrokenProvenance):
                return transform
            if transform is not None:
                reverse_transforms.append(transform)
            current = next_cursor
        else:
            return BrokenProvenance(
                "sampler lineage exceeded the hop limit", node_id=current.link[0]
            )
        if not reverse_stages:
            return BrokenProvenance("sampler lineage contains no supported sampler")
        return AttentionSamplerLineage(tuple(reversed(reverse_stages)))


def _sampling_stage(
    prompt: Mapping[str, Any],
    cursor: _Cursor,
    class_type: str,
    inputs: Mapping[str, Any],
    node_registry: NodeRegistry,
) -> AttentionSamplerStage | BrokenProvenance | None:
    """Resolve recognized direct, guider, and detailer sampling authorities."""

    if class_type in _DIRECT_SAMPLERS:
        return _direct_stage(
            prompt,
            cursor.link[0],
            inputs,
            "latent_image",
            "latent",
            node_registry,
        )
    if class_type == _GUIDER_SAMPLER:
        guider_link = parse_graph_link(inputs.get("guider"))
        upstream = parse_graph_link(inputs.get("latent_image"))
        if guider_link is None or upstream is None:
            return BrokenProvenance(
                "advanced sampler guider or latent input is not a graph link",
                node_id=cursor.link[0],
            )
        guider = _node(prompt, guider_link[0])
        if guider is None:
            return BrokenProvenance(
                "advanced sampler guider is missing", guider_link[0]
            )
        guider_inputs = _inputs(guider)
        model = parse_graph_link(guider_inputs.get("model"))
        positive = parse_graph_link(
            guider_inputs.get("positive") or guider_inputs.get("conditioning")
        )
        if model is None or positive is None:
            return BrokenProvenance(
                "advanced guider MODEL or positive is not a graph link",
                node_id=guider_link[0],
            )
        return AttentionSamplerStage(
            cursor.link[0],
            guider_link[0],
            model,
            positive,
            upstream,
            "latent",
            source_aspect=_source_aspect(
                prompt,
                upstream,
                "latent",
                node_registry,
            ),
        )
    if class_type in _FULL_CANVAS_DETAILERS:
        return _direct_stage(
            prompt,
            cursor.link[0],
            inputs,
            "image",
            "image",
            node_registry,
        )
    if class_type in _CROP_LOCAL_DETAILERS:
        stage = _direct_stage(
            prompt,
            cursor.link[0],
            inputs,
            "image",
            "image",
            node_registry,
        )
        if isinstance(stage, BrokenProvenance):
            return stage
        return replace(
            stage,
            capture_supported=False,
            unsupported_reason=(
                "the sampler runs on crop-local images whose placement is not "
                "available to attention capture"
            ),
        )
    generic = _generic_latent_stage(
        prompt,
        cursor.link[0],
        inputs,
        node_registry,
    )
    if generic is not None:
        return generic
    return None


def _generic_latent_stage(
    prompt: Mapping[str, Any],
    node_id: str,
    inputs: Mapping[str, Any],
    node_registry: NodeRegistry,
) -> AttentionSamplerStage | None:
    """Recognize sampler-compatible nodes by their standard graph inputs."""

    if not all(name in inputs for name in ("model", "positive", "latent_image")):
        return None
    stage = _direct_stage(
        prompt,
        node_id,
        inputs,
        "latent_image",
        "latent",
        node_registry,
    )
    return None if isinstance(stage, BrokenProvenance) else stage


def _direct_stage(
    prompt: Mapping[str, Any],
    node_id: str,
    inputs: Mapping[str, Any],
    upstream_name: str,
    upstream_kind: str,
    node_registry: NodeRegistry,
) -> AttentionSamplerStage | BrokenProvenance:
    """Resolve one node whose MODEL and positive are direct graph inputs."""

    model = parse_graph_link(inputs.get("model"))
    positive = parse_graph_link(inputs.get("positive"))
    upstream = parse_graph_link(inputs.get(upstream_name))
    if model is None or positive is None or upstream is None:
        return BrokenProvenance(
            "sampling stage MODEL, positive, or spatial input is not a graph link",
            node_id=node_id,
        )
    return AttentionSamplerStage(
        node_id,
        node_id,
        model,
        positive,
        upstream,
        upstream_kind,
        source_aspect=_source_aspect(
            prompt,
            upstream,
            upstream_kind,
            node_registry,
        ),
    )


def _spatial_predecessor(
    cursor: _Cursor,
    class_type: str,
    inputs: Mapping[str, Any],
    node_registry: NodeRegistry,
) -> _Cursor | BrokenProvenance:
    """Follow one known modality boundary, resize, or declared passthrough."""

    mapping: Mapping[str, str]
    next_kind = cursor.kind
    if cursor.kind == "image" and class_type in _IMAGE_TO_LATENT:
        mapping = _IMAGE_TO_LATENT
        next_kind = "latent"
    elif cursor.kind == "latent" and class_type in _LATENT_TO_IMAGE:
        mapping = _LATENT_TO_IMAGE
        next_kind = "image"
    elif cursor.kind == "image" and class_type in _IMAGE_TRANSFORMS:
        mapping = _IMAGE_TRANSFORMS
    elif cursor.kind == "latent" and class_type in _LATENT_TRANSFORMS:
        mapping = _LATENT_TRANSFORMS
    else:
        class_def = node_registry.get(class_type)
        rules = (
            getattr(class_def, "GRAPH_PASSTHROUGH_OUTPUTS", None) if class_def else None
        )
        input_name = rules.get(cursor.link[1]) if isinstance(rules, Mapping) else None
        if not isinstance(input_name, str):
            return BrokenProvenance(
                "spatial source has no recognized provenance adapter",
                node_id=cursor.link[0],
                class_type=class_type,
            )
        mapping = {class_type: input_name}
    next_link = parse_graph_link(inputs.get(mapping[class_type]))
    if next_link is None:
        return BrokenProvenance(
            "spatial predecessor input is not a graph link",
            node_id=cursor.link[0],
            class_type=class_type,
        )
    return _Cursor(next_link, next_kind)


def _is_spatial_terminal(inputs: Mapping[str, Any]) -> bool:
    """Return whether a source exposes no earlier image or latent graph link."""

    return not any(
        parse_graph_link(inputs.get(name)) is not None
        for name in ("image", "pixels", "samples", "latent", "latent_image")
    )


def _source_aspect(
    prompt: Mapping[str, Any],
    start_link: GraphLink,
    source_kind: str,
    node_registry: NodeRegistry,
) -> float | None:
    """Resolve a sampler input aspect from graph-visible spatial ancestry."""

    current = _Cursor(start_link, source_kind)
    visited: set[GraphLink] = set()
    for _hop in range(MAX_PROVENANCE_HOPS):
        if current.link in visited:
            return None
        visited.add(current.link)
        node = _node(prompt, current.link[0])
        if node is None:
            return None
        class_type = _class_type(node)
        inputs = _inputs(node)
        explicit = _explicit_output_aspect(class_type, inputs)
        if explicit is not None:
            return explicit
        sampling_input = _sampling_spatial_input(class_type, inputs)
        if sampling_input is not None:
            current = sampling_input
            continue
        predecessor = _spatial_predecessor(
            current,
            class_type,
            inputs,
            node_registry,
        )
        if isinstance(predecessor, BrokenProvenance):
            return None
        current = predecessor
    return None


def _explicit_output_aspect(
    class_type: str,
    inputs: Mapping[str, Any],
) -> float | None:
    """Return dimensions declared by a terminal or sized transform."""

    sized_transform = class_type in {
        "ImageScale",
        "LatentUpscale",
        "SimpleSyrup.ResizeImageToTarget",
    }
    if not sized_transform and not _is_spatial_terminal(inputs):
        return None
    width = inputs.get("width")
    height = inputs.get("height")
    if type(width) is not int or width < 1 or type(height) is not int or height < 1:
        return None
    return width / height


def _sampling_spatial_input(
    class_type: str,
    inputs: Mapping[str, Any],
) -> _Cursor | None:
    """Follow through a sampler while resolving the aspect of a later stage."""

    if class_type in _DIRECT_SAMPLERS or class_type == _GUIDER_SAMPLER:
        link = parse_graph_link(inputs.get("latent_image"))
        return _Cursor(link, "latent") if link is not None else None
    if class_type in _FULL_CANVAS_DETAILERS | _CROP_LOCAL_DETAILERS:
        link = parse_graph_link(inputs.get("image"))
        return _Cursor(link, "image") if link is not None else None
    if all(name in inputs for name in ("model", "positive", "latent_image")):
        link = parse_graph_link(inputs.get("latent_image"))
        return _Cursor(link, "latent") if link is not None else None
    return None


def _node(prompt: Mapping[str, Any], node_id: str) -> Mapping[str, Any] | None:
    """Return one serialized prompt node."""

    value = prompt.get(node_id)
    return value if isinstance(value, Mapping) else None


def _class_type(node: Mapping[str, Any]) -> str:
    """Return one serialized class type."""

    value = node.get("class_type")
    return value if isinstance(value, str) else ""


def _inputs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return serialized node inputs."""

    value = node.get("inputs")
    return value if isinstance(value, Mapping) else {}


ATTENTION_SAMPLER_LINEAGE_RESOLVER = AttentionSamplerLineageResolver()


def _spatial_transform(
    class_type: str, inputs: Mapping[str, Any]
) -> AttentionSpatialTransform | BrokenProvenance | None:
    """Decode one recognized full-canvas transform from serialized inputs."""

    if class_type in {"ImageScaleBy", "LatentUpscaleBy"}:
        return _scale_transform(class_type, inputs.get("scale_by"))
    if class_type in {"ImageScale", "LatentUpscale"}:
        return _sized_transform(
            class_type,
            inputs,
            AttentionSpatialTransformKind.RESIZE,
        )
    if class_type != "SimpleSyrup.ResizeImageToTarget":
        return None
    raw_mode = inputs.get("resize_mode", "Keep AR")
    mode = str(raw_mode).casefold()
    kind = (
        AttentionSpatialTransformKind.FIT_RESIZE
        if "keep ar" in mode
        else AttentionSpatialTransformKind.RESIZE
    )
    if "crop" in mode:
        kind = AttentionSpatialTransformKind.COVER_CROP
    elif "pad" in mode:
        kind = AttentionSpatialTransformKind.FIT_PAD
    result = _sized_transform(class_type, inputs, kind)
    if isinstance(result, BrokenProvenance):
        return result
    divisible_by = inputs.get("divisible_by", 1)
    if type(divisible_by) is not int or divisible_by < 1:
        return BrokenProvenance(
            "spatial divisibility is not graph-visible",
            class_type=class_type,
        )
    return replace(
        result,
        anchor=str(inputs.get("crop_position", "center")),
        divisible_by=divisible_by,
    )


def _scale_transform(
    class_type: str, raw_scale: object
) -> AttentionSpatialTransform | BrokenProvenance:
    """Decode one positive numeric scale factor."""

    if isinstance(raw_scale, bool) or not isinstance(raw_scale, int | float):
        return BrokenProvenance(
            "spatial scale is not a graph-visible number", class_type=class_type
        )
    try:
        return AttentionSpatialTransform(
            AttentionSpatialTransformKind.SCALE,
            scale=float(raw_scale),
        )
    except ValueError as exc:
        return BrokenProvenance(str(exc), class_type=class_type)


def _sized_transform(
    class_type: str,
    inputs: Mapping[str, Any],
    kind: AttentionSpatialTransformKind,
) -> AttentionSpatialTransform | BrokenProvenance:
    """Decode one graph-visible positive target size."""

    width = inputs.get("width")
    height = inputs.get("height")
    if type(width) is not int or type(height) is not int:
        return BrokenProvenance(
            "spatial target size is not graph-visible", class_type=class_type
        )
    try:
        return AttentionSpatialTransform(kind, width=width, height=height)
    except ValueError as exc:
        return BrokenProvenance(str(exc), class_type=class_type)
