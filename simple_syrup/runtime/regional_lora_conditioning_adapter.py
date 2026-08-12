# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt regional conditioning HookGroups into one canonical LoRA plan."""

from __future__ import annotations

from comfy.hooks import HookGroup, WeightHook

from ..domain.raw_regional_attention import (
    RawRegionalAttentionBranch,
    RawRegionalAttentionContext,
    RawRegionalAttentionPlan,
)
from ..domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraBranch,
)
from .regional_lora_plan_adapter import (
    RegionalLoraHookSource,
    RegionalLoraPlanAdaptation,
    RegionalLoraPlanAdapter,
)
from .regional_model_hook_selection import (
    REGIONAL_MODEL_HOOK_SELECTOR,
    RegionalModelHookSelection,
    RegionalModelHookSelector,
)


class RegionalLoraConditioningAdapter:
    """Extract one uniform regional HookGroup per branch and mask position."""

    def __init__(
        self,
        plan_adapter: RegionalLoraPlanAdapter | None = None,
        hook_selector: RegionalModelHookSelector | None = None,
    ) -> None:
        """Retain the authoritative HookGroup-to-domain adapter."""

        self._hook_selector = hook_selector or REGIONAL_MODEL_HOOK_SELECTOR
        self._plan_adapter = plan_adapter or RegionalLoraPlanAdapter(
            self._hook_selector
        )

    def adapt(
        self,
        plan: RawRegionalAttentionPlan,
        *,
        model: object,
    ) -> RegionalLoraPlanAdaptation:
        """Return ordered positive then negative regional adapter uses."""

        if not isinstance(plan, RawRegionalAttentionPlan):
            raise TypeError("Regional LoRA conditioning adaptation requires a plan.")
        self._require_unhooked_base("positive", plan.positive)
        self._require_unhooked_base("negative", plan.negative)
        sources = (
            *self._branch_sources(
                plan.positive,
                branch=RegionalLoraBranch.POSITIVE,
            ),
            *self._branch_sources(
                plan.negative,
                branch=RegionalLoraBranch.NEGATIVE,
            ),
        )
        return self._plan_adapter.adapt(sources, model=model)

    def _branch_sources(
        self,
        branch_plan: RawRegionalAttentionBranch,
        *,
        branch: RegionalLoraBranch,
    ) -> tuple[RegionalLoraHookSource, ...]:
        """Extract canonical HookGroups in one branch's region order."""

        sources: list[RegionalLoraHookSource] = []
        for context in branch_plan.regional_contexts:
            hooks = self._uniform_hooks(context, branch=branch)
            if hooks is None:
                continue
            selection = self._model_hook_selection(
                hooks,
                source_label=(
                    f"Attention Coupling {branch.value} region {context.region_index}"
                ),
            )
            identities = tuple(
                self._identity(
                    participant.hook,
                    branch=branch,
                    context=context,
                    hook_index=participant.hook_index,
                )
                for participant in selection.model_hooks
            )
            sources.append(
                RegionalLoraHookSource(
                    region_index=context.region_index,
                    branch=branch,
                    hooks=hooks,
                    adapter_identities=identities,
                )
            )
        return tuple(sources)

    def _require_unhooked_base(
        self,
        branch_name: str,
        branch: RawRegionalAttentionBranch,
    ) -> None:
        """Require only model-active global LoRAs to arrive on the input MODEL."""

        groups = _conditioning_hook_groups(branch.base_conditioning)
        model_hook_count = sum(
            len(
                self._model_hook_selection(
                    group,
                    source_label=(
                        f"Attention Coupling {branch_name} global conditioning"
                    ),
                ).model_hooks
            )
            for group in groups
        )
        if model_hook_count:
            raise ValueError(
                f"Attention Coupling {branch_name} global conditioning contains "
                "model hooks. Apply global LoRAs to the input MODEL; reserve "
                "conditioning hooks for masked regional entries."
            )

    def _uniform_hooks(
        self,
        context: RawRegionalAttentionContext,
        *,
        branch: RegionalLoraBranch,
    ) -> HookGroup | None:
        """Return one schedule-bearing HookGroup shared by all text entries."""

        groups = _conditioning_hook_groups(context.conditioning)
        if not groups:
            return None
        selections = tuple(
            self._model_hook_selection(
                group,
                source_label=(
                    f"Attention Coupling {branch.value} region "
                    f"{context.region_index} conditioning entry {entry_index}"
                ),
            )
            for entry_index, group in enumerate(groups)
        )
        if not any(selection.model_hooks for selection in selections):
            return None
        authority = groups[0]
        authority_signature = self._group_signature(selections[0])
        if any(
            self._group_signature(selection) != authority_signature
            for selection in selections[1:]
        ):
            raise ValueError(
                f"Attention Coupling {branch.value} region {context.region_index} "
                "uses different model HookGroups across text schedule entries. "
                "Keep model LoRA scheduling on one shared WeightHook schedule."
            )
        return authority

    @staticmethod
    def _group_signature(
        selection: RegionalModelHookSelection,
    ) -> tuple[tuple[object, ...], ...]:
        """Describe hook identity and schedule without decoding adapter payloads."""

        return tuple(
            (
                type(hook),
                getattr(hook, "hook_ref", None),
                getattr(hook, "hook_id", None),
                getattr(hook, "_strength_model", None),
                id(getattr(hook, "weights", None)),
                tuple(
                    (
                        getattr(keyframe, "start_percent", None),
                        getattr(keyframe, "strength", None),
                        getattr(keyframe, "guarantee_steps", None),
                    )
                    for keyframe in getattr(
                        getattr(hook, "hook_keyframe", None),
                        "keyframes",
                        (),
                    )
                ),
            )
            for participant in selection.model_hooks
            for hook in (participant.hook,)
        )

    def _model_hook_selection(
        self,
        hooks: HookGroup,
        *,
        source_label: str,
    ) -> RegionalModelHookSelection:
        """Delegate model participation to its authoritative host adapter."""

        return self._hook_selector.select(hooks, source_label=source_label)

    @staticmethod
    def _identity(
        hook: object,
        *,
        branch: RegionalLoraBranch,
        context: RawRegionalAttentionContext,
        hook_index: int,
    ) -> RegionalLoraAdapterIdentity:
        """Read the stable identity supplied by Prompt Control or the host hook."""

        if not isinstance(hook, WeightHook):
            raise TypeError("Regional LoRA identity requires a Comfy WeightHook.")
        for value in (hook.hook_ref, hook.hook_id):
            if isinstance(value, str) and value.strip():
                return RegionalLoraAdapterIdentity(value)
        raise ValueError(
            f"Attention Coupling {branch.value} region {context.region_index} "
            f"WeightHook {hook_index} has no stable string adapter identity. "
            "Author regional LoRAs through Prompt Control so each hook retains "
            "its LoRA name."
        )


def _conditioning_hook_groups(conditioning: object) -> tuple[HookGroup, ...]:
    """Return every present HookGroup from one standard conditioning value."""

    if not isinstance(conditioning, list) or not conditioning:
        raise TypeError("Regional LoRA conditioning must be a non-empty list.")
    groups: list[HookGroup] = []
    for item_index, item in enumerate(conditioning):
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not isinstance(item[1], dict)
        ):
            raise ValueError(
                f"Regional LoRA conditioning item {item_index} must contain "
                "a context tensor and metadata dictionary."
            )
        hooks = item[1].get("hooks")
        if hooks is None:
            continue
        if not isinstance(hooks, HookGroup):
            raise TypeError(
                f"Regional LoRA conditioning item {item_index} hooks must be a "
                "Comfy HookGroup."
            )
        groups.append(hooks)
    return tuple(groups)


REGIONAL_LORA_CONDITIONING_ADAPTER = RegionalLoraConditioningAdapter()
