# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify Comfy normalized operations and their source-key evidence."""

from __future__ import annotations

from collections.abc import Mapping

from comfy.weight_adapter.base import WeightAdapterBase
from comfy.weight_adapter.lora import LoRAAdapter

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from .comfy_adapter_identity_index import ComfyAdapterIdentityIndex
from .comfy_adapter_resolution import (
    ComfyAdapterResolutionIssue,
    ComfyAdapterResolutionIssueCode,
    ComfyAdapterSourceEntry,
    ComfyAdapterSourceScope,
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
)


def normalized_targets(
    adapter: RegionalLoraAdapterPlan,
    model_weights: Mapping[object, object],
) -> tuple[
    tuple[ComfyNormalizedAdapterTarget, ...],
    tuple[ComfyAdapterResolutionIssue, ...],
]:
    """Classify normalized model operations while preserving their identities."""

    targets: list[ComfyNormalizedAdapterTarget] = []
    issues: list[ComfyAdapterResolutionIssue] = []
    for raw_path, operation in model_weights.items():
        path = _target_path(raw_path)
        if path is None:
            issues.append(
                issue(
                    adapter,
                    ComfyAdapterResolutionIssueCode.INVALID_TARGET_PATH,
                    f"Comfy returned invalid model target path {raw_path!r}.",
                )
            )
            continue
        source_keys = _operation_source_keys(operation)
        is_ordinary = (
            isinstance(operation, LoRAAdapter) and operation.weights[4] is None
        )
        if isinstance(operation, LoRAAdapter) and operation.weights[4] is not None:
            issues.append(
                issue(
                    adapter,
                    ComfyAdapterResolutionIssueCode.DORA_UNSUPPORTED,
                    f"Target {path.parameter_key!r} uses unsupported DoRA scaling.",
                )
            )
        elif not isinstance(operation, LoRAAdapter):
            issues.append(
                issue(
                    adapter,
                    ComfyAdapterResolutionIssueCode.UNSUPPORTED_OPERATION,
                    f"Target {path.parameter_key!r} uses unsupported "
                    f"{type(operation).__name__} operation.",
                )
            )
        targets.append(
            ComfyNormalizedAdapterTarget(
                path=path,
                operation=operation,
                operation_type=type(operation).__name__,
                source_keys=source_keys,
                ordinary_additive_lora=is_ordinary,
            )
        )
    return tuple(targets), tuple(issues)


def classify_raw_sources(
    raw_weights: Mapping[str, object],
    *,
    model_weights: Mapping[object, object],
    clip_weights: Mapping[object, object],
    vae_weights: Mapping[object, object],
) -> tuple[ComfyAdapterSourceEntry, ...]:
    """Classify every raw key from Comfy's normalized payload evidence."""

    model_keys = _normalized_source_keys(model_weights, raw_weights)
    clip_keys = _normalized_source_keys(clip_weights, raw_weights)
    vae_keys = _normalized_source_keys(vae_weights, raw_weights)
    return tuple(
        ComfyAdapterSourceEntry(
            source_key=source_key,
            scope=(
                ComfyAdapterSourceScope.MODEL
                if source_key in model_keys
                else ComfyAdapterSourceScope.TEXT_ENCODER
                if source_key in clip_keys
                else ComfyAdapterSourceScope.VAE
                if source_key in vae_keys
                else ComfyAdapterSourceScope.UNUSED
            ),
        )
        for source_key in raw_weights
    )


def classify_initialized_sources(
    *,
    model_weights: Mapping[object, object],
    clip_weights: Mapping[object, object],
) -> tuple[ComfyAdapterSourceEntry, ...]:
    """Classify source keys exposed by already-normalized host operations."""

    model_keys = _operation_mapping_source_keys(model_weights)
    clip_keys = _operation_mapping_source_keys(clip_weights)
    return tuple(
        ComfyAdapterSourceEntry(source_key, ComfyAdapterSourceScope.MODEL)
        for source_key in sorted(model_keys)
    ) + tuple(
        ComfyAdapterSourceEntry(source_key, ComfyAdapterSourceScope.TEXT_ENCODER)
        for source_key in sorted(clip_keys - model_keys)
    )


def source_keys(value: object | None) -> tuple[str, ...]:
    """Return ordered string source keys when a failed payload exposes them."""

    if not isinstance(value, Mapping):
        return ()
    return tuple(key for key in value if isinstance(key, str))


def issue(
    adapter: RegionalLoraAdapterPlan,
    code: ComfyAdapterResolutionIssueCode,
    message: str,
) -> ComfyAdapterResolutionIssue:
    """Create one consistently adapter-scoped issue."""

    return ComfyAdapterResolutionIssue(
        composition_index=adapter.composition_index,
        adapter_identity=adapter.adapter_identity.value,
        code=code,
        message=message,
    )


def _normalized_source_keys(
    normalized: Mapping[object, object],
    raw_weights: Mapping[str, object],
) -> set[str]:
    """Recover consumed keys from adapter metadata or retained value identity."""

    exposed = _operation_mapping_source_keys(normalized)
    identities = ComfyAdapterIdentityIndex.build(normalized.values())
    return exposed | {
        source_key
        for source_key, source_value in raw_weights.items()
        if identities.contains(source_value)
    }


def _operation_mapping_source_keys(weights: Mapping[object, object]) -> set[str]:
    """Collect source keys exposed by installed normalized adapter objects."""

    return {
        source_key
        for operation in weights.values()
        for source_key in _operation_source_keys(operation)
    }


def _operation_source_keys(operation: object) -> tuple[str, ...]:
    """Read exact source keys exposed by one installed Comfy adapter."""

    if not isinstance(operation, WeightAdapterBase):
        return ()
    return tuple(sorted(operation.loaded_keys))


def _target_path(value: object) -> ComfyAdapterTargetPath | None:
    """Retain string and sliced target paths exactly as Comfy emits them."""

    if isinstance(value, str):
        return ComfyAdapterTargetPath(value, None)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], tuple)
        and all(
            isinstance(part, int) and not isinstance(part, bool) for part in value[1]
        )
    ):
        return ComfyAdapterTargetPath(value[0], value[1])
    return None
