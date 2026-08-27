# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observe selected cross-attention affinities without replacing denoising."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import RLock

import torch

from ..domain.attention_geometry import factor_spatial_geometry
from ..domain.attention_region_capture import (
    AttentionCapturePlan,
    AttentionCaptureProfile,
    AttentionEvidenceMode,
    AttentionRegionRequest,
)
from ..domain.attention_region_maps import (
    AttentionTokenCatalog,
    AttentionTokenSpan,
    CapturedAttentionMap,
    OpenVocabularyContext,
)
from ..domain.regional_model_capabilities import RegionalModelFamily
from .attention_coupling.anima_context import ANIMA_CONTEXT_SEQUENCE_LENGTH
from .attention_region_affinity import (
    ATTENTION_AFFINITY_CALCULATOR,
    PendingAttentionMap,
)
from .attention_region_cadence import AttentionCaptureCadence
from .attention_region_self_completion import (
    MAXIMUM_SELF_COMPLETION_TOKENS,
    SpatialSelfAttention,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RequestTargets:
    """Bind one request to exact prompt spans selected for capture."""

    request: AttentionRegionRequest
    spans: tuple[AttentionTokenSpan, ...]


@dataclass(frozen=True, slots=True)
class _PendingSelfAttention:
    """Retain one spatial self-attention projection until its paired cross call."""

    query: torch.Tensor
    key: torch.Tensor
    heads: int
    skip_reshape: bool
    branches: object


class AttentionRegionCaptureSession:
    """Aggregate compact selected-token affinity maps for one sampler run."""

    def __init__(
        self,
        *,
        plan: AttentionCapturePlan,
        model_family: RegionalModelFamily,
        token_catalog: AttentionTokenCatalog,
        request_spans: Mapping[str, tuple[AttentionTokenSpan, ...]],
        open_vocabulary_contexts: tuple[OpenVocabularyContext, ...] = (),
    ) -> None:
        """Resolve request targets and initialize prompt-scoped observations."""

        self.plan = plan
        self.model_family = model_family
        self.token_catalog = token_catalog
        self.open_vocabulary_contexts = open_vocabulary_contexts
        self._targets = tuple(
            _RequestTargets(request, request_spans.get(request.node_id, ()))
            for request in plan.requests
        )
        self._pending_maps: list[PendingAttentionMap] = []
        self._materialized_maps: tuple[CapturedAttentionMap, ...] | None = None
        self._pending_open_keys: tuple[tuple[str, torch.Tensor], ...] = ()
        self._pending_self_attention: dict[str, _PendingSelfAttention] = {}
        self._captured_call_index = 0
        self._geometry_logged = False
        self._geometry_unavailable = False
        self._lock = RLock()
        profiles = tuple(target.request.controls.profile for target in self._targets)
        self._stride = min(_profile_stride(profile) for profile in profiles)
        self._cadence = AttentionCaptureCadence(self._stride)
        self._self_completion_stride = min(
            _profile_self_completion_stride(profile) for profile in profiles
        )
        self._denominator_token_budget = max(
            _profile_denominator_tokens(profile) for profile in profiles
        )
        self._capture_start = min(
            target.request.controls.capture_start for target in self._targets
        )
        self._capture_end = max(
            target.request.controls.capture_end for target in self._targets
        )

    @property
    def status_message(self) -> str:
        """Return a concise model-family capture status for public nodes."""

        if self._geometry_unavailable:
            return (
                "No attention maps: the selected Anima sampler input dimensions "
                "are not graph-visible"
            )
        return f"Captured {self.model_family.value} attention"

    def observe(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        transformer_options: Mapping[str, object],
        *,
        skip_reshape: bool,
    ) -> None:
        """Capture selected positive-branch affinities for one attention call."""

        if not any(target.spans for target in self._targets) and not (
            self.open_vocabulary_contexts
        ):
            return
        if not _has_expected_cross_attention_context(
            key,
            self.model_family,
            self.token_catalog.sequence_length,
            skip_reshape,
        ):
            self._stage_self_attention(
                query,
                key,
                heads,
                transformer_options,
                skip_reshape=skip_reshape,
            )
            return
        open_keys = self._take_open_vocabulary_keys()
        layer_key = _layer_key(transformer_options)
        pending_self_attention = self._take_self_attention(layer_key)
        step_index = self._cadence.sample(layer_key)
        if step_index is None:
            return
        progress = self._cadence.progress(step_index, transformer_options)
        if progress < self._capture_start or progress > self._capture_end:
            return
        if not self._has_resolvable_geometry(transformer_options):
            return
        q_heads, k_heads = ATTENTION_AFFINITY_CALCULATOR.head_tensors(
            query, key, heads, skip_reshape
        )
        v_heads = ATTENTION_AFFINITY_CALCULATOR.head_tensor(value, heads, skip_reshape)
        q_positive, k_positive, v_positive = (
            ATTENTION_AFFINITY_CALCULATOR.positive_rows(
                q_heads,
                k_heads,
                v_heads,
                transformer_options.get("cond_or_uncond"),
            )
        )
        spans = _unique_spans(self._targets)
        derive_concept_values = self.model_family is RegionalModelFamily.STANDARD_UNET
        contextual_spans = (
            _contextual_spans(self._targets) if derive_concept_values else ()
        )
        denominator = ATTENTION_AFFINITY_CALCULATOR.log_denominator(
            q_positive,
            k_positive,
            self._denominator_token_budget,
        )
        spatial_height, spatial_width = _spatial_geometry(
            int(q_positive.shape[-2]),
            transformer_options,
            fallback_aspect=self.plan.source_aspect,
        )
        self._log_geometry_once(
            token_count=int(q_positive.shape[-2]),
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            transformer_options=transformer_options,
        )
        captured = ATTENTION_AFFINITY_CALCULATOR.capture_spans(
            q_positive,
            k_positive,
            v_positive,
            spans,
            progress,
            layer_key,
            denominator,
            spatial_height,
            spatial_width,
            contextual_targets=contextual_spans,
            context_candidates=self.token_catalog.spans,
            self_attention=(
                self._prepare_self_attention(
                    pending_self_attention,
                    expected_batch=int(q_positive.shape[0]),
                    expected_tokens=int(q_positive.shape[-2]),
                )
                if self._use_self_completion()
                else None
            ),
            derive_concept_values=derive_concept_values,
        )
        open_captured = tuple(
            captured_map
            for label, projected_key in open_keys
            for captured_map in ATTENTION_AFFINITY_CALCULATOR.capture_open_vocabulary(
                q_positive,
                projected_key,
                heads,
                label,
                progress,
                layer_key,
                k_positive,
                v_positive,
                self._denominator_token_budget,
                spatial_height,
                spatial_width,
            )
        )
        with self._lock:
            self._pending_maps.extend((*captured, *open_captured))

    def _stage_self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        heads: int,
        transformer_options: Mapping[str, object],
        *,
        skip_reshape: bool,
    ) -> None:
        """Stage bounded SDXL spatial projections for concept isolation."""

        if (
            self.model_family is not RegionalModelFamily.STANDARD_UNET
            or not _contextual_spans(self._targets)
            or not _is_bounded_self_attention(query, key, skip_reshape)
        ):
            return
        pending = _PendingSelfAttention(
            query=query,
            key=key,
            heads=heads,
            skip_reshape=skip_reshape,
            branches=transformer_options.get("cond_or_uncond"),
        )
        with self._lock:
            self._pending_self_attention[_layer_key(transformer_options)] = pending

    def _take_self_attention(
        self,
        layer_key: str,
    ) -> _PendingSelfAttention | None:
        """Consume only a self-attention projection paired with this layer and step."""

        with self._lock:
            return self._pending_self_attention.pop(layer_key, None)

    @staticmethod
    def _prepare_self_attention(
        pending: _PendingSelfAttention | None,
        *,
        expected_batch: int,
        expected_tokens: int,
    ) -> SpatialSelfAttention | None:
        """Normalize and select the positive rows of staged spatial projections."""

        if pending is None:
            return None
        query, key = ATTENTION_AFFINITY_CALCULATOR.head_tensors(
            pending.query,
            pending.key,
            pending.heads,
            pending.skip_reshape,
        )
        query, key, _unused = ATTENTION_AFFINITY_CALCULATOR.positive_rows(
            query,
            key,
            key,
            pending.branches,
        )
        if (
            int(query.shape[0]) != expected_batch
            or int(query.shape[-2]) != expected_tokens
        ):
            return None
        return SpatialSelfAttention(query=query, key=key)

    def _use_self_completion(self) -> bool:
        """Subsample expensive spatial completion according to capture quality."""

        with self._lock:
            index = self._captured_call_index
            self._captured_call_index += 1
        return index % self._self_completion_stride == 0

    def _has_resolvable_geometry(
        self,
        transformer_options: Mapping[str, object],
    ) -> bool:
        """Fail closed when an Anima spatial grid has no trustworthy orientation."""

        if self.model_family is not RegionalModelFamily.ANIMA:
            return True
        if (
            self.plan.source_aspect is not None
            or _runtime_aspect(transformer_options) is not None
        ):
            return True
        with self._lock:
            first_failure = not self._geometry_unavailable
            self._geometry_unavailable = True
        if first_failure:
            LOGGER.warning(
                "Skipped Anima attention capture because the selected sampler "
                "input dimensions are not graph-visible",
                extra={"sampler_node_id": self.plan.sampler_node_id},
            )
        return False

    def _log_geometry_once(
        self,
        *,
        token_count: int,
        spatial_height: int,
        spatial_width: int,
        transformer_options: Mapping[str, object],
    ) -> None:
        """Log one representative geometry decision for runtime diagnosis."""

        with self._lock:
            if self._geometry_logged:
                return
            self._geometry_logged = True
        original_shape = transformer_options.get("original_shape")
        activations_shape = transformer_options.get("activations_shape")
        LOGGER.info(
            "Resolved %s attention capture geometry: %d tokens -> %dx%d; "
            "original_shape=%r activations_shape=%r",
            self.model_family.value,
            token_count,
            spatial_height,
            spatial_width,
            original_shape,
            activations_shape,
            extra={
                "model_family": self.model_family.value,
                "spatial_token_count": token_count,
                "spatial_height": spatial_height,
                "spatial_width": spatial_width,
                "original_shape": original_shape,
                "activations_shape": activations_shape,
            },
        )

    def stage_open_vocabulary_keys(
        self,
        keys: tuple[tuple[str, torch.Tensor], ...],
    ) -> None:
        """Stage keys produced by the current layer's clone-local wrapper."""

        with self._lock:
            self._pending_open_keys = keys

    def _take_open_vocabulary_keys(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Consume only keys paired with the immediately following attention call."""

        with self._lock:
            keys = self._pending_open_keys
            self._pending_open_keys = ()
            return keys

    def maps_for(self, request_node_id: str) -> tuple[CapturedAttentionMap, ...]:
        """Return immutable maps whose labels belong to one planned request."""

        target = next(
            (
                candidate
                for candidate in self._targets
                if candidate.request.node_id == request_node_id
            ),
            None,
        )
        if target is None:
            raise KeyError(f"Unknown attention-region request node: {request_node_id}.")
        labels = {span.display_label for span in target.spans}
        labels.update(
            context.label
            for context in self.open_vocabulary_contexts
            if context.label in target.request.queries
        )
        maps = self._materialize_maps()
        return tuple(
            replace(value, spatial_transforms=target.request.spatial_transforms)
            for value in maps
            if value.label in labels
        )

    def _materialize_maps(self) -> tuple[CapturedAttentionMap, ...]:
        """Materialize shared observations exactly once across sibling consumers."""

        with self._lock:
            if self._materialized_maps is not None:
                return self._materialized_maps
            pending = tuple(self._pending_maps)
            self._pending_maps.clear()
            self._materialized_maps = ATTENTION_AFFINITY_CALCULATOR.materialize(
                pending,
                self.model_family,
            )
            return self._materialized_maps


def _profile_stride(profile: AttentionCaptureProfile) -> int:
    """Return deterministic attention-call subsampling for one capture profile."""

    if profile is AttentionCaptureProfile.FAST:
        return 32
    if profile is AttentionCaptureProfile.BALANCED:
        return 4
    return 1


def _profile_denominator_tokens(profile: AttentionCaptureProfile) -> int:
    """Bound probability calibration work according to the selected profile."""

    if profile is AttentionCaptureProfile.FAST:
        return 8
    if profile is AttentionCaptureProfile.BALANCED:
        return 16
    return 512


def _profile_self_completion_stride(profile: AttentionCaptureProfile) -> int:
    """Return the profile's model-native spatial completion cadence."""

    if profile is AttentionCaptureProfile.FAST:
        return 16
    if profile is AttentionCaptureProfile.BALANCED:
        return 2
    return 1


def _unique_spans(
    targets: tuple[_RequestTargets, ...],
) -> tuple[AttentionTokenSpan, ...]:
    """Deduplicate coalesced target spans while preserving prompt order."""

    return tuple(dict.fromkeys(span for target in targets for span in target.spans))


def _contextual_spans(
    targets: tuple[_RequestTargets, ...],
) -> tuple[AttentionTokenSpan, ...]:
    """Return spans requested by at least one concept-isolation consumer."""

    return tuple(
        dict.fromkeys(
            span
            for target in targets
            if target.request.controls.evidence_mode is AttentionEvidenceMode.CONCEPT
            for span in target.spans
        )
    )


def _layer_key(options: Mapping[str, object]) -> str:
    """Return a stable diagnostic key from Comfy transformer block metadata."""

    block = options.get("block", "unknown")
    block_index = options.get("block_index", 0)
    transformer_index = options.get("transformer_index", 0)
    return f"{block!r}:{block_index!r}:{transformer_index!r}"


def _spatial_geometry(
    token_count: int,
    options: Mapping[str, object],
    *,
    fallback_aspect: float | None,
) -> tuple[int, int]:
    """Resolve the captured attention grid from Comfy's source tensor shape."""

    target_aspect = _runtime_aspect(options) or fallback_aspect or 1.0
    return factor_spatial_geometry(token_count, target_aspect=target_aspect)


def _runtime_aspect(options: Mapping[str, object]) -> float | None:
    """Return a trustworthy aspect from installed transformer metadata."""

    original_shape = options.get("original_shape")
    if not isinstance(original_shape, (tuple, list)) or len(original_shape) < 2:
        return None
    raw_height, raw_width = original_shape[-2:]
    if (
        type(raw_height) is not int
        or raw_height < 1
        or type(raw_width) is not int
        or raw_width < 1
    ):
        return None
    return raw_width / raw_height


def _has_expected_cross_attention_context(
    key: torch.Tensor,
    model_family: RegionalModelFamily,
    mapped_token_count: int,
    skip_reshape: bool,
) -> bool:
    """Reject self-attention calls before capture cadence and affinity work."""

    if key.ndim not in (3, 4):
        return False
    context_tokens = int(
        key.shape[-2] if skip_reshape and key.ndim == 4 else key.shape[1]
    )
    expected = (
        ANIMA_CONTEXT_SEQUENCE_LENGTH
        if model_family is RegionalModelFamily.ANIMA
        else mapped_token_count
    )
    return context_tokens == expected


def _is_bounded_self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    skip_reshape: bool,
) -> bool:
    """Return whether an attention call is a safe spatial self-attention source."""

    expected_rank = 4 if skip_reshape else 3
    if query.ndim != expected_rank or key.ndim != expected_rank:
        return False
    query_tokens = int(query.shape[-2] if skip_reshape else query.shape[1])
    key_tokens = int(key.shape[-2] if skip_reshape else key.shape[1])
    return 1 < query_tokens == key_tokens <= MAXIMUM_SELF_COMPLETION_TOKENS
