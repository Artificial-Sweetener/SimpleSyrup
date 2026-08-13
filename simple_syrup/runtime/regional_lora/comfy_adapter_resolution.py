# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable evidence returned by Comfy regional adapter resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from ..regional_lora_host_payload import RegionalLoraHostPayload


class ComfyAdapterSourceScope(StrEnum):
    """Classify one source entry using installed host-resolution evidence."""

    MODEL = "model"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"
    UNRESOLVED = "unresolved"
    UNUSED = "unused"


class ComfyAdapterResolutionIssueCode(StrEnum):
    """Identify one adapter-scoped failure without discarding other results."""

    RESOLUTION_FAILED = "resolution_failed"
    INVALID_TARGET_PATH = "invalid_target_path"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    DORA_UNSUPPORTED = "dora_unsupported"


@dataclass(frozen=True, slots=True)
class ComfyAdapterTargetPath:
    """Retain Comfy's exact model parameter key and optional tensor slice."""

    parameter_key: str
    offset: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ComfyAdapterSourceEntry:
    """Retain one raw source key and its host-proven resolution scope."""

    source_key: str
    scope: ComfyAdapterSourceScope


@dataclass(frozen=True, slots=True)
class ComfyNormalizedAdapterTarget:
    """Retain one exact normalized Comfy operation by identity."""

    path: ComfyAdapterTargetPath
    operation: object
    operation_type: str
    source_keys: tuple[str, ...]
    ordinary_additive_lora: bool


@dataclass(frozen=True, slots=True)
class ComfyAdapterResolutionIssue:
    """Describe one failure in the context of its canonical adapter use."""

    composition_index: int
    adapter_identity: str
    code: ComfyAdapterResolutionIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class ComfyRegionalAdapterResolution:
    """Retain all resolution evidence for one ordered regional adapter use."""

    adapter: RegionalLoraAdapterPlan
    payload: RegionalLoraHostPayload
    model_targets: tuple[ComfyNormalizedAdapterTarget, ...]
    source_entries: tuple[ComfyAdapterSourceEntry, ...]
    issues: tuple[ComfyAdapterResolutionIssue, ...]


@dataclass(frozen=True, slots=True)
class ComfyRegionalLoraResolution:
    """Retain ordered adapter results and every aggregated issue."""

    adapters: tuple[ComfyRegionalAdapterResolution, ...]
    issues: tuple[ComfyAdapterResolutionIssue, ...]
