# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt standard-UNet calls to shared schedule and CFG invariants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingCallHarness,
    AttentionCouplingCallObservation,
    AttentionCouplingCallScenario,
)
from comfy.patcher_extension import WrappersMP

from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
)
from simple_syrup.runtime.attention_coupling.unet_attention_context_wrapper import (
    UNET_ATTENTION_CONTEXT_WRAPPER_KEY,
    StandardUnetAttentionContextDiffusionWrapper,
)
from simple_syrup.runtime.attention_coupling.unet_attention_phase_session import (
    STANDARD_UNET_ATTENTION_PHASE_SESSION,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)


class _UnetDiffusionModel(torch.nn.Module):
    """Provide weak-referenceable UNet model identity for call invariants."""


@dataclass
class _UnetCallExecutor:
    """Observe dynamic UNet state inside one nested model call."""

    class_obj: object
    state: StandardUnetAttentionState
    calls: int = 0
    observed: BatchedRegionalAttentionContexts | None = None

    def __call__(self, *_: object, **__: object) -> torch.Tensor:
        """Capture active contexts and return one model-shaped result."""

        self.calls += 1
        self.observed = self.state.execution_context.require_current()
        return torch.zeros(1)


class UnetAttentionCouplingCallHarness(AttentionCouplingCallHarness):
    """Resolve shared model-call state through the production UNet wrapper."""

    def resolve_call(
        self,
        scenario: AttentionCouplingCallScenario,
    ) -> AttentionCouplingCallObservation:
        """Resolve one scheduled CFG call through the UNet context wrapper."""

        plan = scenario.plan
        state = StandardUnetAttentionState(
            plan,
            (1.0,) * plan.mask_bank.region_count,
            RegionalAttentionDiagnosticsBuilder(
                plan.mask_bank,
                backend="invariant.standard-unet",
            ),
        )
        model = _UnetDiffusionModel()
        wrapper = StandardUnetAttentionContextDiffusionWrapper(
            state,
            STANDARD_UNET_ATTENTION_PHASE_SESSION,
        )
        executor = _UnetCallExecutor(model, state)
        base_context = _scheduled_base_context(scenario)
        base_before = base_context.clone()
        wrapper(
            executor,
            torch.zeros(len(scenario.selectors), 4, 1, 1),
            torch.full((len(scenario.selectors),), scenario.sigma),
            base_context,
            None,
            None,
            {
                "cond_or_uncond": list(scenario.selectors),
                "sample_sigmas": torch.tensor([scenario.sigma, 0.0]),
                "sigmas": torch.full(
                    (len(scenario.selectors),),
                    scenario.sigma,
                ),
                "wrappers": {
                    WrappersMP.DIFFUSION_MODEL: {
                        UNET_ATTENTION_CONTEXT_WRAPPER_KEY: [wrapper]
                    }
                },
            },
        )
        if executor.observed is None:
            raise AssertionError("UNet call wrapper did not publish contexts")
        state_restored = False
        try:
            state.execution_context.require_current()
        except RuntimeError:
            state_restored = True
        return _call_observation(
            executor.observed,
            calls=executor.calls,
            source_context_unchanged=torch.equal(base_context, base_before),
            state_restored=state_restored,
            canonical_plan_preserved=state.plan is plan
            and state.diagnostics.mask_bank is plan.mask_bank,
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
    """Narrow dynamic UNet contexts to shared observable values."""

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
