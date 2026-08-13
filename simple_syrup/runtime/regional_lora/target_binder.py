# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind resolved regional LoRA targets to one effective Comfy model graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import comfy.model_patcher

from ...domain.resolved_regional_lora import (
    ResolvedRegionalLoraOperation,
    ResolvedRegionalLoraOperationSet,
)
from .comfy_adapter_resolution import (
    ComfyNormalizedAdapterTarget,
    ComfyRegionalLoraResolution,
)
from .effective_model_graph import (
    EFFECTIVE_MODEL_GRAPH_RESOLVER,
    EffectiveModelGraphError,
    EffectiveModelGraphResolver,
)
from .target_binding import (
    BoundRegionalLoraModuleClass,
    BoundRegionalLoraOperation,
    BoundRegionalLoraSpatialCapability,
    RegionalLoraBindingIssue,
    RegionalLoraBindingIssueCode,
    RegionalLoraBindingResult,
)
from .target_shape_admission import (
    REGIONAL_LORA_TARGET_SHAPE_VALIDATOR,
    RegionalLoraTargetShapeValidator,
)

LOGGER = logging.getLogger(__name__)


class RegionalLoraTargetBinder:
    """Bind every target atomically in evidence while never mutating the graph."""

    def __init__(
        self,
        graph_resolver: EffectiveModelGraphResolver = EFFECTIVE_MODEL_GRAPH_RESOLVER,
        shape_validator: RegionalLoraTargetShapeValidator = (
            REGIONAL_LORA_TARGET_SHAPE_VALIDATOR
        ),
    ) -> None:
        """Retain focused effective-graph and shape-admission collaborators."""

        self._graph_resolver = graph_resolver
        self._shape_validator = shape_validator

    def bind(
        self,
        *,
        source: object,
        candidate: object,
        resolution: ComfyRegionalLoraResolution,
        operations: ResolvedRegionalLoraOperationSet,
        linear_spatial_capabilities: Mapping[
            str,
            BoundRegionalLoraSpatialCapability,
        ]
        | None = None,
    ) -> RegionalLoraBindingResult:
        """Return complete ordered bindings or aggregate pre-mutation failures."""

        capabilities = _linear_spatial_capabilities(linear_spatial_capabilities)
        lineage_issue = _lineage_issue(source, candidate)
        if lineage_issue is not None:
            return RegionalLoraBindingResult((), (lineage_issue,))
        assert isinstance(candidate, comfy.model_patcher.ModelPatcher)
        pairing_issues, pairs = _ordered_pairs(resolution, operations)
        duplicate_keys = _duplicate_ownership_keys(operations.entries)
        entries: list[BoundRegionalLoraOperation] = []
        for descriptor, normalized_target in pairs:
            entry = self._bind_target(
                candidate,
                descriptor,
                normalized_target,
                linear_spatial_capabilities=capabilities,
                duplicate=(
                    descriptor.adapter.composition_index,
                    descriptor.target.parameter_path,
                    descriptor.target.offset,
                )
                in duplicate_keys,
            )
            entries.append(entry)
        resolution_issues = tuple(
            RegionalLoraBindingIssue(
                composition_index=issue.composition_index,
                target_index=None,
                parameter_path=None,
                code=RegionalLoraBindingIssueCode.RESOLUTION_ISSUE,
                message=issue.message,
            )
            for issue in resolution.issues
        )
        issues = (
            pairing_issues
            + resolution_issues
            + tuple(issue for entry in entries for issue in entry.issues)
        )
        return RegionalLoraBindingResult(tuple(entries), issues)

    def _bind_target(
        self,
        candidate: comfy.model_patcher.ModelPatcher,
        descriptor: ResolvedRegionalLoraOperation,
        normalized_target: ComfyNormalizedAdapterTarget,
        *,
        linear_spatial_capabilities: Mapping[
            str,
            BoundRegionalLoraSpatialCapability,
        ],
        duplicate: bool,
    ) -> BoundRegionalLoraOperation:
        """Bind one paired target and retain every local failure."""

        issues: list[RegionalLoraBindingIssue] = []
        if duplicate:
            issues.append(
                _issue(
                    descriptor,
                    RegionalLoraBindingIssueCode.DUPLICATE_OWNERSHIP,
                    "One adapter owns the same target path and offset more than once.",
                )
            )
        if not descriptor.supported:
            issues.append(
                _issue(
                    descriptor,
                    RegionalLoraBindingIssueCode.OPERATION_REJECTED,
                    descriptor.rejection_reason or "Resolved operation is rejected.",
                )
            )
            return _unbound(descriptor, normalized_target, tuple(issues))
        try:
            effective_module = self._graph_resolver.resolve(
                candidate,
                descriptor.target.model_target,
            )
            effective_parameter = self._graph_resolver.resolve(
                candidate,
                descriptor.target.parameter_path,
            )
        except EffectiveModelGraphError as error:
            code = (
                RegionalLoraBindingIssueCode.OBJECT_PATCH_CONFLICT
                if "overlapping object patch" in str(error)
                else RegionalLoraBindingIssueCode.TARGET_NOT_FOUND
            )
            issues.append(_issue(descriptor, code, str(error)))
            return _unbound(descriptor, normalized_target, tuple(issues))
        admission = self._shape_validator.validate(
            descriptor,
            effective_module.value,
            effective_parameter.value,
        )
        if admission.issue is not None:
            issues.append(
                _issue(
                    descriptor,
                    _admission_issue_code(admission.issue),
                    admission.issue,
                )
            )
        capability = _spatial_capability(
            descriptor.target.parameter_path,
            admission.module_class,
            linear_spatial_capabilities,
        )
        return BoundRegionalLoraOperation(
            descriptor=descriptor,
            normalized_target=normalized_target,
            module=effective_module.value,
            parameter=admission.parameter,
            module_class=admission.module_class,
            spatial_capability=capability,
            active_object_patch_paths=tuple(
                dict.fromkeys(
                    effective_module.active_object_patch_paths
                    + effective_parameter.active_object_patch_paths
                )
            ),
            issues=tuple(issues),
        )


def _lineage_issue(
    source: object, candidate: object
) -> RegionalLoraBindingIssue | None:
    """Require the source itself or one direct Comfy clone sharing its model."""

    if not isinstance(source, comfy.model_patcher.ModelPatcher) or not isinstance(
        candidate, comfy.model_patcher.ModelPatcher
    ):
        return _global_issue(
            RegionalLoraBindingIssueCode.SOURCE_LINEAGE,
            "Regional LoRA target binding requires Comfy ModelPatchers.",
        )
    if candidate is source:
        return None
    if candidate.parent is not source or not source.is_clone(candidate):
        return _global_issue(
            RegionalLoraBindingIssueCode.SOURCE_LINEAGE,
            "Regional LoRA candidate must be the source or its direct Comfy clone.",
        )
    return None


def _ordered_pairs(
    resolution: ComfyRegionalLoraResolution,
    operations: ResolvedRegionalLoraOperationSet,
) -> tuple[
    tuple[RegionalLoraBindingIssue, ...],
    tuple[tuple[ResolvedRegionalLoraOperation, ComfyNormalizedAdapterTarget], ...],
]:
    """Pair exact U2 identities with U3 descriptors without positional guessing."""

    normalized = {
        (adapter.adapter.composition_index, target_index): target
        for adapter in resolution.adapters
        for target_index, target in enumerate(adapter.model_targets)
    }
    descriptors = {
        (entry.adapter.composition_index, entry.target_index): entry
        for entry in operations.entries
    }
    issues: list[RegionalLoraBindingIssue] = []
    missing_descriptors = sorted(normalized.keys() - descriptors.keys())
    missing_targets = sorted(descriptors.keys() - normalized.keys())
    for key in missing_descriptors:
        issues.append(
            _global_issue(
                RegionalLoraBindingIssueCode.RESOLUTION_MISMATCH,
                f"Normalized adapter/target {key} has no U3 descriptor.",
            )
        )
    for key in missing_targets:
        issues.append(
            _global_issue(
                RegionalLoraBindingIssueCode.RESOLUTION_MISMATCH,
                f"U3 adapter/target {key} has no normalized operation identity.",
            )
        )
    pairs = tuple(
        (
            descriptor,
            normalized[(descriptor.adapter.composition_index, descriptor.target_index)],
        )
        for descriptor in operations.entries
        if (descriptor.adapter.composition_index, descriptor.target_index) in normalized
    )
    return tuple(issues), pairs


def _duplicate_ownership_keys(
    descriptors: tuple[ResolvedRegionalLoraOperation, ...],
) -> set[tuple[int, str, tuple[int, ...] | None]]:
    """Find repeated ownership within one adapter while allowing composition stacks."""

    seen: set[tuple[int, str, tuple[int, ...] | None]] = set()
    duplicates: set[tuple[int, str, tuple[int, ...] | None]] = set()
    for descriptor in descriptors:
        key = (
            descriptor.adapter.composition_index,
            descriptor.target.parameter_path,
            descriptor.target.offset,
        )
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _spatial_capability(
    parameter_path: str,
    module_class: BoundRegionalLoraModuleClass,
    linear_capabilities: Mapping[str, BoundRegionalLoraSpatialCapability],
) -> BoundRegionalLoraSpatialCapability:
    """Publish only spatial capability directly proven by the observed operation."""

    if module_class in (
        BoundRegionalLoraModuleClass.CONVOLUTION_1D,
        BoundRegionalLoraModuleClass.CONVOLUTION_2D,
        BoundRegionalLoraModuleClass.CONVOLUTION_3D,
    ):
        return BoundRegionalLoraSpatialCapability.DIRECT
    if module_class is BoundRegionalLoraModuleClass.LINEAR:
        return linear_capabilities.get(
            parameter_path,
            BoundRegionalLoraSpatialCapability.GLOBAL_ONLY,
        )
    return BoundRegionalLoraSpatialCapability.UNSUPPORTED


def _linear_spatial_capabilities(
    capabilities: Mapping[str, BoundRegionalLoraSpatialCapability] | None,
) -> Mapping[str, BoundRegionalLoraSpatialCapability]:
    """Validate explicit consumer-spatialized declarations without path policy."""

    if capabilities is None:
        return {}
    for path, capability in capabilities.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Linear spatial capability paths must be nonempty.")
        if capability not in (
            BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
            BoundRegionalLoraSpatialCapability.PACKED_IMAGE_TOKENS,
            BoundRegionalLoraSpatialCapability.PACKED_CONTEXT_TOKENS,
            BoundRegionalLoraSpatialCapability.GLOBAL_ONLY,
        ):
            raise ValueError(
                "Linear target capability must declare one supported token role or "
                "global-only ownership."
            )
    return capabilities


def _admission_issue_code(message: str) -> RegionalLoraBindingIssueCode:
    """Map focused admission text into stable binder issue categories."""

    if "module type" in message:
        return RegionalLoraBindingIssueCode.MODULE_UNSUPPORTED
    if "no parameter" in message:
        return RegionalLoraBindingIssueCode.PARAMETER_NOT_FOUND
    if "parameter must" in message:
        return RegionalLoraBindingIssueCode.PARAMETER_INVALID
    return RegionalLoraBindingIssueCode.SHAPE_INCOMPATIBLE


def _issue(
    descriptor: ResolvedRegionalLoraOperation,
    code: RegionalLoraBindingIssueCode,
    message: str,
) -> RegionalLoraBindingIssue:
    """Create one target-scoped issue."""

    return RegionalLoraBindingIssue(
        composition_index=descriptor.adapter.composition_index,
        target_index=descriptor.target_index,
        parameter_path=descriptor.target.parameter_path,
        code=code,
        message=message,
    )


def _global_issue(
    code: RegionalLoraBindingIssueCode,
    message: str,
) -> RegionalLoraBindingIssue:
    """Create one bind-wide issue without inventing target ownership."""

    return RegionalLoraBindingIssue(None, None, None, code, message)


def _unbound(
    descriptor: ResolvedRegionalLoraOperation,
    normalized_target: ComfyNormalizedAdapterTarget,
    issues: tuple[RegionalLoraBindingIssue, ...],
) -> BoundRegionalLoraOperation:
    """Return one unresolved target without partial module claims."""

    return BoundRegionalLoraOperation(
        descriptor=descriptor,
        normalized_target=normalized_target,
        module=None,
        parameter=None,
        module_class=BoundRegionalLoraModuleClass.UNSUPPORTED,
        spatial_capability=BoundRegionalLoraSpatialCapability.UNSUPPORTED,
        active_object_patch_paths=(),
        issues=issues,
    )


REGIONAL_LORA_TARGET_BINDER = RegionalLoraTargetBinder()
