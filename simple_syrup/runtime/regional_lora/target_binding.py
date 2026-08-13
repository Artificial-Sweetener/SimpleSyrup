# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable runtime evidence for regional LoRA target binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.resolved_regional_lora import ResolvedRegionalLoraOperation
from .comfy_adapter_resolution import ComfyNormalizedAdapterTarget


class BoundRegionalLoraModuleClass(StrEnum):
    """Classify the observed executable module owning one target parameter."""

    LINEAR = "linear"
    CONVOLUTION_1D = "convolution_1d"
    CONVOLUTION_2D = "convolution_2d"
    CONVOLUTION_3D = "convolution_3d"
    UNSUPPORTED = "unsupported"


class BoundRegionalLoraSpatialCapability(StrEnum):
    """Declare how one bound operation can receive regional spatial ownership."""

    DIRECT = "direct"
    SPATIAL_TOKENS = "spatial_tokens"
    PACKED_IMAGE_TOKENS = "packed_image_tokens"
    PACKED_CONTEXT_TOKENS = "packed_context_tokens"
    GLOBAL_ONLY = "global_only"
    UNSUPPORTED = "unsupported"


class RegionalLoraBindingIssueCode(StrEnum):
    """Identify one complete-bind failure without discarding sibling evidence."""

    SOURCE_LINEAGE = "source_lineage"
    RESOLUTION_MISMATCH = "resolution_mismatch"
    RESOLUTION_ISSUE = "resolution_issue"
    DUPLICATE_OWNERSHIP = "duplicate_ownership"
    OBJECT_PATCH_CONFLICT = "object_patch_conflict"
    TARGET_NOT_FOUND = "target_not_found"
    MODULE_UNSUPPORTED = "module_unsupported"
    PARAMETER_NOT_FOUND = "parameter_not_found"
    PARAMETER_INVALID = "parameter_invalid"
    SHAPE_INCOMPATIBLE = "shape_incompatible"
    OPERATION_REJECTED = "operation_rejected"


@dataclass(frozen=True, slots=True)
class RegionalLoraBindingIssue:
    """Describe one adapter/target-scoped binding failure."""

    composition_index: int | None
    target_index: int | None
    parameter_path: str | None
    code: RegionalLoraBindingIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class BoundRegionalLoraOperation:
    """Pair U2/U3 evidence with exact observed module and parameter identities."""

    descriptor: ResolvedRegionalLoraOperation
    normalized_target: ComfyNormalizedAdapterTarget
    module: object | None
    parameter: object | None
    module_class: BoundRegionalLoraModuleClass
    spatial_capability: BoundRegionalLoraSpatialCapability
    active_object_patch_paths: tuple[str, ...]
    issues: tuple[RegionalLoraBindingIssue, ...]

    @property
    def bound(self) -> bool:
        """Report whether this target has one executable, issue-free binding."""

        return (
            not self.issues
            and self.module is not None
            and self.parameter is not None
            and self.module_class is not BoundRegionalLoraModuleClass.UNSUPPORTED
            and self.spatial_capability
            is not BoundRegionalLoraSpatialCapability.UNSUPPORTED
        )


@dataclass(frozen=True, slots=True)
class RegionalLoraBindingResult:
    """Retain all ordered target evidence and aggregate all binding failures."""

    entries: tuple[BoundRegionalLoraOperation, ...]
    issues: tuple[RegionalLoraBindingIssue, ...]

    @property
    def admissible(self) -> bool:
        """Require every target to bind and every aggregate issue to be absent."""

        return not self.issues and all(entry.bound for entry in self.entries)
