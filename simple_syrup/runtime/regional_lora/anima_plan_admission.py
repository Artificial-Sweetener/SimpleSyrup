# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admit every ordered regional Anima adapter before execution construction."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan, RegionalLoraPlan
from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from .anima_targets import (
    ANIMA_LORA_TARGET_CLASSIFIER,
    AnimaLoraAdmission,
    AnimaLoraAdmissionError,
    AnimaLoraTargetClassifier,
)


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraPlanAdmissionIssue:
    """Identify one adapter-scoped format or target admission failure."""

    composition_index: int
    adapter_identity: str
    key: str
    reason: str


class AnimaRegionalLoraPlanAdmissionError(ValueError):
    """Report every invalid adapter without returning a partial plan."""

    def __init__(
        self,
        issues: tuple[AnimaRegionalLoraPlanAdmissionIssue, ...],
    ) -> None:
        """Create one actionable ordered aggregate diagnostic."""

        self.issues = issues
        details = "\n".join(
            f"- adapter {issue.composition_index} {issue.adapter_identity!r} key "
            f"{issue.key!r}: {issue.reason}"
            for issue in issues
        )
        super().__init__(
            "Regional Anima LoRA plan admission failed before sampling:\n" + details
        )


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraAdapterAdmission:
    """Bind one canonical plan entry to its completely admitted target set."""

    adapter_plan: RegionalLoraAdapterPlan
    admission: AnimaLoraAdmission

    def __post_init__(self) -> None:
        """Require exact plan and admission value types."""

        if not isinstance(self.adapter_plan, RegionalLoraAdapterPlan):
            raise TypeError("Anima regional LoRA adapter plan has an invalid type.")
        if not isinstance(self.admission, AnimaLoraAdmission):
            raise TypeError("Anima regional LoRA target admission has an invalid type.")


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraPlanAdmission:
    """Retain one all-or-nothing admission aligned to the immutable plan."""

    plan: RegionalLoraPlan
    adapters: tuple[AnimaRegionalLoraAdapterAdmission, ...]

    def __post_init__(self) -> None:
        """Require canonical identity-preserving alignment with the source plan."""

        if not isinstance(self.plan, RegionalLoraPlan):
            raise TypeError("Anima regional LoRA admission requires a plan.")
        if not isinstance(self.adapters, tuple):
            raise TypeError("Anima regional LoRA admissions must be a tuple.")
        if len(self.adapters) != len(self.plan.adapters):
            raise ValueError("Anima regional LoRA admissions must align with the plan.")
        if any(
            not isinstance(adapter, AnimaRegionalLoraAdapterAdmission)
            for adapter in self.adapters
        ):
            raise TypeError("Anima regional LoRA admission contains an invalid entry.")
        if any(
            admitted.adapter_plan is not planned
            for admitted, planned in zip(
                self.adapters,
                self.plan.adapters,
                strict=True,
            )
        ):
            raise ValueError(
                "Anima regional LoRA admissions must retain exact plan entries."
            )


class AnimaRegionalLoraPlanAdmissionService:
    """Apply Anima adapter policy to a complete ordered regional plan."""

    def __init__(
        self,
        classifier: AnimaLoraTargetClassifier = ANIMA_LORA_TARGET_CLASSIFIER,
    ) -> None:
        """Retain the sole individual-adapter classifier dependency."""

        self._classifier = classifier

    def admit(
        self,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AnimaRegionalLoraPlanAdmission:
        """Reject every invalid adapter before returning any admitted plan."""

        if not isinstance(adaptation, RegionalLoraPlanAdaptation):
            raise TypeError("Regional Anima LoRA admission requires an adaptation.")
        plan = adaptation.plan

        admitted: list[AnimaRegionalLoraAdapterAdmission] = []
        issues: list[AnimaRegionalLoraPlanAdmissionIssue] = []
        for adapter_plan, weights in zip(
            plan.adapters,
            (payload.model_side_weights for payload in adaptation.adapter_payloads),
            strict=True,
        ):
            try:
                admission = self._classifier.admit(weights)
            except AnimaLoraAdmissionError as error:
                issues.extend(
                    AnimaRegionalLoraPlanAdmissionIssue(
                        composition_index=adapter_plan.composition_index,
                        adapter_identity=adapter_plan.adapter_identity.value,
                        key=issue.key,
                        reason=issue.reason,
                    )
                    for issue in error.issues
                )
                continue
            admitted.append(AnimaRegionalLoraAdapterAdmission(adapter_plan, admission))
        if issues:
            raise AnimaRegionalLoraPlanAdmissionError(tuple(issues))
        return AnimaRegionalLoraPlanAdmission(plan, tuple(admitted))


ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE = AnimaRegionalLoraPlanAdmissionService()
