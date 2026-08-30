# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan and inject prompt-scoped attention capture from downstream graph intent."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, replace
from typing import Any

from ..domain.attention_concepts import parse_attention_concepts
from ..domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionCaptureProfile,
    AttentionEvidenceMode,
    AttentionRegionControls,
    AttentionRegionRequest,
    AttentionRegionRequestKind,
)
from ..domain.attention_sampler_lineage import AttentionSamplerStage
from ..domain.graph_provenance import BrokenProvenance, GraphLink
from .attention_sampler_lineage import ATTENTION_SAMPLER_LINEAGE_RESOLVER
from .comfy_graph_provenance import (
    NodeRegistry,
    parse_graph_link,
)

LOGGER = logging.getLogger(__name__)
INTERNAL_CAPTURE_NODE_ID = "SimpleSyrup.AttentionCaptureModel"
REQUEST_NODE_KINDS: dict[str, AttentionRegionRequestKind] = {
    "SimpleSyrup.ConceptAttentionSEGS": AttentionRegionRequestKind.CONCEPT_SEGS,
    "SimpleSyrup.AllPromptAttentionSEGS": AttentionRegionRequestKind.ALL_PROMPT_SEGS,
    "SimpleSyrup.AttentionRegionMask": AttentionRegionRequestKind.REGION_MASK,
    "SimpleSyrup.AttentionMaskedConditioning": (
        AttentionRegionRequestKind.MASKED_CONDITIONING
    ),
}


class AttentionRegionPromptPlanner:
    """Resolve public attention nodes into one immutable plan per sampler."""

    def build(
        self,
        prompt: Mapping[str, Any],
        node_registry: NodeRegistry,
    ) -> tuple[AttentionCapturePlan, ...]:
        """Return ordered coalesced plans for every recoverable request node."""

        grouped: dict[
            str, list[tuple[AttentionRegionRequest, AttentionSamplerStage]]
        ] = defaultdict(list)
        for node_id in sorted(prompt):
            node = _node(prompt, node_id)
            if node is None:
                continue
            kind = REQUEST_NODE_KINDS.get(_class_type(node))
            if kind is None:
                continue
            request = _request(node_id, kind, node)
            source = self._resolve_source(prompt, node, node_registry)
            if isinstance(source, BrokenProvenance):
                LOGGER.warning(
                    "Attention-region graph request cannot recover a sampler",
                    extra={"request_node_id": node_id, "reason": source.reason},
                )
                continue
            request = replace(request, spatial_transforms=source.forward_transforms)
            grouped[source.sampler_node_id].append((request, source))

        plans: list[AttentionCapturePlan] = []
        for sampler_id in sorted(grouped):
            pairs = grouped[sampler_id]
            authority = pairs[0][1]
            if any(
                (
                    source.model_owner_node_id,
                    source.model_link,
                    source.positive_link,
                )
                != (
                    authority.model_owner_node_id,
                    authority.model_link,
                    authority.positive_link,
                )
                for _request_value, source in pairs[1:]
            ):
                LOGGER.warning(
                    "Attention-region requests disagree about sampler authority",
                    extra={"sampler_node_id": sampler_id},
                )
                continue
            requests = tuple(
                sorted(
                    (request for request, _source in pairs),
                    key=lambda value: value.node_id,
                )
            )
            prompt_text, clip_link = _trace_prompt_text(prompt, authority.positive_link)
            plans.append(
                AttentionCapturePlan(
                    sampler_node_id=authority.sampler_node_id,
                    model_owner_node_id=authority.model_owner_node_id,
                    model_input_name="model",
                    model_link=authority.model_link,
                    positive_link=authority.positive_link,
                    requests=requests,
                    prompt_text=prompt_text,
                    clip_link=clip_link,
                    source_aspect=authority.source_aspect,
                )
            )
        return tuple(plans)

    def _resolve_source(
        self,
        prompt: Mapping[str, Any],
        request_node: Mapping[str, Any],
        node_registry: NodeRegistry,
    ) -> AttentionSamplerStage | BrokenProvenance:
        """Select one stage from the request's complete sampler lineage."""

        inputs = _inputs(request_node)
        source_name = "image" if "image" in inputs else "latent"
        source_link = parse_graph_link(inputs.get(source_name))
        if source_link is None:
            return BrokenProvenance(f"{source_name} input is not a graph link")
        lineage = ATTENTION_SAMPLER_LINEAGE_RESOLVER.resolve(
            prompt=prompt,
            start_link=source_link,
            source_kind=source_name,
            node_registry=node_registry,
        )
        if isinstance(lineage, BrokenProvenance):
            return lineage
        requested_stage = int(inputs.get("sampler_stage", 1))
        selection = lineage.select(requested_stage)
        if selection.was_clamped:
            LOGGER.info(
                "Attention sampler stage was clamped to direct provenance",
                extra={
                    "requested_stage": requested_stage,
                    "selected_stage": selection.stage_number,
                    "stage_count": selection.stage_count,
                },
            )
        if not selection.stage.capture_supported:
            return BrokenProvenance(
                selection.stage.unsupported_reason
                or "selected sampler stage cannot project attention to its output",
                node_id=selection.stage.sampler_node_id,
            )
        return selection.stage


class AttentionRegionPromptRewriter:
    """Inject one internal MODEL patch node for each planned sampler."""

    def rewrite(
        self,
        prompt: MutableMapping[str, Any],
        plans: tuple[AttentionCapturePlan, ...],
    ) -> None:
        """Mutate the prompt graph without changing unrelated node inputs."""

        for plan in plans:
            if plan.capture_node_id in prompt:
                raise ValueError(
                    "Attention capture node id collides with the prompt graph."
                )
            owner = _node(prompt, plan.model_owner_node_id)
            if owner is None:
                raise ValueError(
                    "Attention capture model owner disappeared during rewrite."
                )
            owner_inputs = _mutable_inputs(owner)
            current_model = parse_graph_link(owner_inputs.get(plan.model_input_name))
            if current_model != plan.model_link:
                raise ValueError(
                    "Attention capture model edge changed during planning."
                )
            capture_inputs: dict[str, Any] = {
                "model": [plan.model_link[0], plan.model_link[1]],
                "positive": [plan.positive_link[0], plan.positive_link[1]],
                "plan_json": _serialize_plan(plan),
            }
            if plan.clip_link is not None:
                capture_inputs["clip"] = [plan.clip_link[0], plan.clip_link[1]]
            prompt[plan.capture_node_id] = {
                "class_type": INTERNAL_CAPTURE_NODE_ID,
                "inputs": capture_inputs,
            }
            owner_inputs[plan.model_input_name] = [plan.capture_node_id, 0]


def _request(
    node_id: str,
    kind: AttentionRegionRequestKind,
    node: Mapping[str, Any],
) -> AttentionRegionRequest:
    """Parse one public node request from serialized prompt inputs."""

    inputs = _inputs(node)
    raw_concepts = inputs.get("concepts", "")
    queries = parse_attention_concepts(
        raw_concepts if isinstance(raw_concepts, str) else str(raw_concepts)
    )
    controls = AttentionRegionControls(
        capture_start=float(inputs.get("capture_start", 0.0)),
        capture_end=float(inputs.get("capture_end", 1.0)),
        minimum_strength=float(inputs.get("minimum_strength", 0.15)),
        minimum_consensus=float(inputs.get("minimum_consensus", 0.25)),
        split_sensitivity=float(inputs.get("split_sensitivity", 0.35)),
        minimum_region_size=int(inputs.get("minimum_region_size", 512)),
        instance_recall=float(inputs.get("instance_recall", 0.65)),
        geometry_recall=float(inputs.get("geometry_recall", 0.85)),
        keep_only=int(inputs.get("keep_only", 1)),
        keep_by=str(inputs.get("keep_by", "largest size")),
        combine_segs=bool(inputs.get("combine_segs", False)),
        matte_solidity=float(inputs.get("matte_solidity", 0.75)),
        edge_feather=int(inputs.get("edge_feather", 8)),
        profile=AttentionCaptureProfile(str(inputs.get("capture_profile", "fast"))),
        evidence_mode=AttentionEvidenceMode(
            str(inputs.get("evidence_mode", AttentionEvidenceMode.CONCEPT.value))
        ),
    )
    return AttentionRegionRequest(
        node_id,
        kind,
        queries,
        controls,
        int(inputs.get("sampler_stage", 1)),
    )


def _trace_prompt_text(
    prompt: Mapping[str, Any],
    positive_link: GraphLink,
) -> tuple[str | None, GraphLink | None]:
    """Recover direct CLIP text and CLIP provenance when graph-visible."""

    node = _node(prompt, positive_link[0])
    if node is None or _class_type(node) != "CLIPTextEncode":
        return None, None
    inputs = _inputs(node)
    text = inputs.get("text")
    clip_link = parse_graph_link(inputs.get("clip"))
    return (text if isinstance(text, str) else None), clip_link


def _serialize_plan(plan: AttentionCapturePlan) -> str:
    """Serialize only capture-affecting controls into the upstream MODEL node."""

    capture_requests = tuple(
        replace(
            request,
            controls=AttentionRegionControls(
                capture_start=request.controls.capture_start,
                capture_end=request.controls.capture_end,
                minimum_strength=0.0,
                minimum_consensus=0.0,
                split_sensitivity=0.0,
                minimum_region_size=1,
                profile=request.controls.profile,
                evidence_mode=request.controls.evidence_mode,
            ),
        )
        for request in plan.requests
    )
    payload = asdict(replace(plan, requests=capture_requests))
    payload["model_link"] = list(plan.model_link)
    payload["positive_link"] = list(plan.positive_link)
    payload["clip_link"] = list(plan.clip_link) if plan.clip_link is not None else None
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _node(prompt: Mapping[str, Any], node_id: str) -> Mapping[str, Any] | None:
    """Return one valid serialized prompt node."""

    value = prompt.get(node_id)
    return value if isinstance(value, Mapping) else None


def _class_type(node: Mapping[str, Any]) -> str:
    """Return a serialized class type or an empty unsupported value."""

    value = node.get("class_type")
    return value if isinstance(value, str) else ""


def _inputs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return immutable serialized inputs or an empty mapping."""

    value = node.get("inputs")
    return value if isinstance(value, Mapping) else {}


def _mutable_inputs(node: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return mutable serialized inputs required for prompt rewriting."""

    value = node.get("inputs")
    if not isinstance(value, MutableMapping):
        raise TypeError("Prompt node inputs must be mutable for attention capture.")
    return value


ATTENTION_REGION_PROMPT_PLANNER = AttentionRegionPromptPlanner()
ATTENTION_REGION_PROMPT_REWRITER = AttentionRegionPromptRewriter()
