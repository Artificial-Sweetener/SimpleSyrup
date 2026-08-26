# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare an observation-only MODEL from one prompt-injected capture plan."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionRegionRequestKind,
)
from ..domain.attention_region_maps import (
    AttentionTokenCatalog,
    AttentionTokenSpan,
    CapturedAttentionMap,
    OpenVocabularyContext,
)
from ..domain.regional_model_capabilities import RegionalModelFamily
from ..runtime.attention_region_capture import AttentionRegionCaptureSession
from ..runtime.attention_region_capture_backend import (
    ATTENTION_REGION_CAPTURE_BACKEND,
)
from ..runtime.attention_region_plan_codec import ATTENTION_CAPTURE_PLAN_CODEC
from ..runtime.attention_region_store import (
    ATTENTION_REGION_CAPTURE_STORE,
    AttentionCaptureSession,
)
from ..runtime.attention_region_tokens import ATTENTION_PROMPT_TOKENIZER
from ..runtime.regional_model_capabilities import RegionalModelCapabilityRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmptyAttentionCaptureSession:
    """Represent an observable unsupported capture without fabricated maps."""

    request_node_ids: tuple[str, ...]
    status_message: str

    def maps_for(self, request_node_id: str) -> tuple[CapturedAttentionMap, ...]:
        """Return no maps after validating request ownership."""

        if request_node_id not in self.request_node_ids:
            raise KeyError(f"Unknown attention-region request node: {request_node_id}.")
        return ()


class AttentionCaptureModelService:
    """Select the model adapter, publish session state, and derive MODEL."""

    def __init__(self) -> None:
        """Create an isolated model capability registry."""

        self._capabilities = RegionalModelCapabilityRegistry()

    def prepare(
        self,
        *,
        model: object,
        plan_json: str,
        clip: object | None,
    ) -> object:
        """Return a capture MODEL or the original MODEL for an observable no-op."""

        plan = ATTENTION_CAPTURE_PLAN_CODEC.decode(plan_json)
        request_ids = tuple(request.node_id for request in plan.requests)
        try:
            capabilities = self._capabilities.capabilities_for(model)
            if clip is None or plan.prompt_text is None:
                raise ValueError("prompt text or CLIP provenance is not graph-visible")
            catalog = ATTENTION_PROMPT_TOKENIZER.build_catalog(
                clip=clip,
                prompt_text=plan.prompt_text,
                model_family=capabilities.model_family,
            )
            request_spans, unresolved = _resolve_request_spans(
                plan=plan,
                clip=clip,
                catalog=catalog,
                model_family=capabilities.model_family,
            )
            open_contexts: tuple[OpenVocabularyContext, ...] = ()
            if capabilities.model_family.value == "standard_unet":
                open_contexts = tuple(
                    ATTENTION_PROMPT_TOKENIZER.encode_open_vocabulary_query(
                        clip=clip,
                        query=query,
                    )
                    for query in unresolved
                )
            capture_session = AttentionRegionCaptureSession(
                plan=plan,
                model_family=capabilities.model_family,
                token_catalog=catalog,
                request_spans=request_spans,
                open_vocabulary_contexts=_unique_open_contexts(open_contexts),
            )
            session: AttentionCaptureSession = capture_session
            derived = ATTENTION_REGION_CAPTURE_BACKEND.derive(model, capture_session)
        except (TypeError, ValueError, RuntimeError) as exc:
            reason = f"Attention capture unavailable: {exc}"
            LOGGER.warning(
                "Attention-region MODEL preparation no-op",
                extra={
                    "sampler_node_id": plan.sampler_node_id,
                    "request_node_ids": request_ids,
                    "reason": str(exc),
                },
            )
            session = EmptyAttentionCaptureSession(request_ids, reason)
            derived = model
        ATTENTION_REGION_CAPTURE_STORE.publish(request_ids, session)
        return derived


ATTENTION_CAPTURE_MODEL_SERVICE = AttentionCaptureModelService()


def _resolve_request_spans(
    *,
    plan: AttentionCapturePlan,
    clip: object,
    catalog: AttentionTokenCatalog,
    model_family: RegionalModelFamily,
) -> tuple[dict[str, tuple[AttentionTokenSpan, ...]], tuple[str, ...]]:
    """Resolve and deduplicate native request spans before MODEL patching."""

    resolved: dict[str, tuple[AttentionTokenSpan, ...]] = {}
    unresolved: list[str] = []
    for request in plan.requests:
        if request.kind is AttentionRegionRequestKind.ALL_PROMPT_SEGS:
            resolved[request.node_id] = catalog.spans
            continue
        spans: list[AttentionTokenSpan] = []
        for query in request.queries:
            matches = ATTENTION_PROMPT_TOKENIZER.resolve_query(
                clip=clip,
                query=query,
                catalog=catalog,
                model_family=model_family,
            )
            if matches:
                spans.extend(matches)
            else:
                unresolved.append(query)
                LOGGER.info(
                    "Attention concept is absent from native conditioning",
                    extra={"request_node_id": request.node_id, "concept": query},
                )
        resolved[request.node_id] = tuple(dict.fromkeys(spans))
    return resolved, tuple(dict.fromkeys(unresolved))


def _unique_open_contexts(
    contexts: tuple[OpenVocabularyContext, ...],
) -> tuple[OpenVocabularyContext, ...]:
    """Deduplicate query contexts by normalized public label in stable order."""

    unique: dict[str, OpenVocabularyContext] = {}
    for context in contexts:
        unique.setdefault(context.label.casefold(), context)
    return tuple(unique.values())
