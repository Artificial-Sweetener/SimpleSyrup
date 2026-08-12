# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare base sampler inputs and validate raw Attention Coupling metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..domain.raw_regional_attention import (
    RawRegionalAttentionBranch,
    RawRegionalAttentionPlan,
)

_UNSUPPORTED_METADATA_KEYS = frozenset(
    {
        "area",
        "control",
        "control_apply_to_uncond",
        "default",
        "gligen",
        "mask",
        "mask_strength",
        "reference_latent",
        "set_area_to_bounds",
    }
)


@dataclass(frozen=True, slots=True)
class AttentionCouplingPreparation:
    """Retain the full raw plan and base-only ordinary sampler inputs."""

    plan: RawRegionalAttentionPlan
    positive: object
    negative: object


@dataclass(frozen=True, slots=True)
class AttentionCouplingMetadataIssue:
    """Identify one unsupported or malformed conditioning metadata field."""

    branch: str
    conditioning_index: int
    item_index: int
    key: str
    reason: str


class AttentionCouplingMetadataError(ValueError):
    """Report every metadata problem before patch installation."""

    def __init__(self, issues: tuple[AttentionCouplingMetadataIssue, ...]) -> None:
        """Create one aggregate actionable diagnostic."""

        self.issues = issues
        details = "\n".join(
            f"- {issue.branch} conditioning {issue.conditioning_index} item "
            f"{issue.item_index} metadata {issue.key!r}: {issue.reason}"
            for issue in issues
        )
        super().__init__(
            "Attention Coupling conditioning metadata is unsupported:\n" + details
        )


class AttentionCouplingPreparationService:
    """Separate regional contexts from ordinary KSampler conditioning."""

    def prepare(self, plan: RawRegionalAttentionPlan) -> AttentionCouplingPreparation:
        """Validate all contexts before returning the untouched base conditionings."""

        if not isinstance(plan, RawRegionalAttentionPlan):
            raise TypeError("Attention Coupling preparation requires a raw plan.")
        issues = self._branch_issues("positive", plan.positive) + self._branch_issues(
            "negative",
            plan.negative,
        )
        if issues:
            raise AttentionCouplingMetadataError(issues)
        return AttentionCouplingPreparation(
            plan=plan,
            positive=plan.positive.base_conditioning,
            negative=plan.negative.base_conditioning,
        )

    def _branch_issues(
        self,
        branch_name: str,
        branch: RawRegionalAttentionBranch,
    ) -> tuple[AttentionCouplingMetadataIssue, ...]:
        """Validate one base plus ordered regional branch completely."""

        indexed_conditionings = ((0, branch.base_conditioning),) + tuple(
            (context.conditioning_index, context.conditioning)
            for context in branch.regional_contexts
        )
        issues: list[AttentionCouplingMetadataIssue] = []
        for conditioning_index, conditioning in indexed_conditionings:
            issues.extend(
                self._conditioning_issues(
                    branch_name,
                    conditioning_index,
                    conditioning,
                )
            )
        return tuple(issues)

    def _conditioning_issues(
        self,
        branch_name: str,
        conditioning_index: int,
        conditioning: object,
    ) -> tuple[AttentionCouplingMetadataIssue, ...]:
        """Validate one standard Comfy conditioning container and its metadata."""

        if not isinstance(conditioning, list) or not conditioning:
            return (
                AttentionCouplingMetadataIssue(
                    branch_name,
                    conditioning_index,
                    -1,
                    "<conditioning>",
                    "must be a non-empty standard CONDITIONING list",
                ),
            )
        issues: list[AttentionCouplingMetadataIssue] = []
        for item_index, item in enumerate(conditioning):
            if not isinstance(item, list | tuple) or len(item) != 2:
                issues.append(
                    AttentionCouplingMetadataIssue(
                        branch_name,
                        conditioning_index,
                        item_index,
                        "<item>",
                        "must contain a context tensor and metadata dictionary",
                    )
                )
                continue
            if not isinstance(item[0], torch.Tensor):
                issues.append(
                    AttentionCouplingMetadataIssue(
                        branch_name,
                        conditioning_index,
                        item_index,
                        "<context>",
                        "must be a torch.Tensor",
                    )
                )
            metadata = item[1]
            if not isinstance(metadata, dict):
                issues.append(
                    AttentionCouplingMetadataIssue(
                        branch_name,
                        conditioning_index,
                        item_index,
                        "<metadata>",
                        "must be a dictionary",
                    )
                )
                continue
            for key in sorted(metadata, key=str):
                if not isinstance(key, str):
                    issues.append(
                        AttentionCouplingMetadataIssue(
                            branch_name,
                            conditioning_index,
                            item_index,
                            repr(key),
                            "metadata keys must be strings",
                        )
                    )
                elif key in _UNSUPPORTED_METADATA_KEYS:
                    issues.append(
                        AttentionCouplingMetadataIssue(
                            branch_name,
                            conditioning_index,
                            item_index,
                            key,
                            "is not supported by Anima Attention Coupling",
                        )
                    )
            issues.extend(
                self._schedule_issues(
                    branch_name,
                    conditioning_index,
                    item_index,
                    metadata,
                )
            )
        return tuple(issues)

    @staticmethod
    def _schedule_issues(
        branch_name: str,
        conditioning_index: int,
        item_index: int,
        metadata: dict[object, object],
    ) -> tuple[AttentionCouplingMetadataIssue, ...]:
        """Validate optional authored conditioning percentage boundaries."""

        issues: list[AttentionCouplingMetadataIssue] = []
        values: dict[str, float] = {}
        for key in ("start_percent", "end_percent"):
            if key not in metadata:
                continue
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                issues.append(
                    AttentionCouplingMetadataIssue(
                        branch_name,
                        conditioning_index,
                        item_index,
                        key,
                        "must be a finite real number in [0, 1]",
                    )
                )
                continue
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                issues.append(
                    AttentionCouplingMetadataIssue(
                        branch_name,
                        conditioning_index,
                        item_index,
                        key,
                        "must be a finite real number in [0, 1]",
                    )
                )
                continue
            values[key] = normalized
        start = values.get("start_percent", 0.0)
        end = values.get("end_percent", 1.0)
        if start > end:
            issues.append(
                AttentionCouplingMetadataIssue(
                    branch_name,
                    conditioning_index,
                    item_index,
                    "<schedule>",
                    "start_percent must not exceed end_percent",
                )
            )
        return tuple(issues)


ATTENTION_COUPLING_PREPARATION_SERVICE = AttentionCouplingPreparationService()
