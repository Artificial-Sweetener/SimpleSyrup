"""Adapt Anima diffusion calls to shared schedule and CFG invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingCallHarness,
    AttentionCouplingCallObservation,
    AttentionCouplingCallScenario,
)
from torch import nn

from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
)
from simple_syrup.runtime.regional_attention_template import (
    build_regional_attention_template,
)
from simple_syrup.runtime.regional_lora.anima_attention_context_wrapper import (
    AnimaRegionalAttentionContextDiffusionWrapper,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface


class _AnimaDiffusionModel(nn.Module):
    """Provide weak-referenceable Anima model identity for call invariants."""


@dataclass
class _AnimaCallExecutor:
    """Observe dynamic Anima state inside one nested model call."""

    class_obj: object
    execution: AnimaRegionalAttentionExecution
    calls: int = 0
    observed: BatchedRegionalAttentionContexts | None = None

    def __call__(self, *_: object, **__: object) -> torch.Tensor:
        """Capture active contexts and return one model-shaped result."""

        self.calls += 1
        self.observed = self.execution.active_contexts
        return torch.zeros(1)


class AnimaAttentionCouplingCallHarness(AttentionCouplingCallHarness):
    """Resolve shared model-call state through the production Anima wrapper."""

    def resolve_call(
        self,
        scenario: AttentionCouplingCallScenario,
    ) -> AttentionCouplingCallObservation:
        """Resolve one scheduled CFG call through the Anima context wrapper."""

        plan = scenario.plan
        template = build_regional_attention_template(plan, latent_batch_size=1)
        execution = AnimaRegionalAttentionExecution(
            template,
            plan.mask_bank,
            (1.0,) * plan.mask_bank.region_count,
            dynamic_contexts=True,
        )
        model = _AnimaDiffusionModel()
        wrapper = AnimaRegionalAttentionContextDiffusionWrapper(
            AnimaModuleSurface(cast(Any, model), (), ()),
            plan,
            execution,
        )
        executor = _AnimaCallExecutor(model, execution)
        base_context = _scheduled_base_context(scenario)
        base_before = base_context.clone()
        wrapper(
            executor,
            torch.zeros(len(scenario.selectors), 16, 1, 1, 1),
            torch.full((len(scenario.selectors),), scenario.sigma),
            base_context,
            None,
            None,
            transformer_options={
                "cond_or_uncond": list(scenario.selectors),
                "sigmas": torch.full(
                    (len(scenario.selectors),),
                    scenario.sigma,
                ),
            },
        )
        if executor.observed is None:
            raise AssertionError("Anima call wrapper did not publish contexts")
        return _call_observation(
            executor.observed,
            calls=executor.calls,
            source_context_unchanged=torch.equal(base_context, base_before),
            state_restored=execution.active_contexts is template,
            canonical_plan_preserved=plan.mask_bank is execution.mask_bank,
        )


def _scheduled_base_context(scenario: AttentionCouplingCallScenario) -> torch.Tensor:
    """Concatenate exact plan-owned base tensors in current CFG order."""

    branches = {
        0: scenario.plan.positive,
        1: scenario.plan.negative,
    }
    return torch.cat(
        tuple(
            branches[selector].base_context.entries[0].cross_attention
            for selector in scenario.selectors
        )
    )


def _call_observation(
    contexts: BatchedRegionalAttentionContexts,
    *,
    calls: int,
    source_context_unchanged: bool,
    state_restored: bool,
    canonical_plan_preserved: bool,
) -> AttentionCouplingCallObservation:
    """Narrow dynamic Anima contexts to shared observable values."""

    return AttentionCouplingCallObservation(
        branches=tuple(chunk.branch for chunk in contexts.chunks),
        base_values=tuple(float(value) for value in contexts.base_context[:, 0, 0]),
        regional_entry_values=tuple(
            tuple(float(value) for value in entry.context[:, 0, 0])
            for region in contexts.regions
            for entry in region.entries
        ),
        nested_model_calls=calls,
        source_context_unchanged=source_context_unchanged,
        state_restored=state_restored,
        canonical_plan_preserved=canonical_plan_preserved,
    )
