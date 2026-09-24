# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify minimum-call persistent standard-UNet regional execution."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.regional_attention_execution_context import (
    RegionalAttentionExecutionContext,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_base_attention import (
    StandardUnetVariantBaseAttention,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_execution import (
    StandardUnetVariantExecution,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_topology import (
    StandardUnetRegionalVariant,
    StandardUnetVariantAdapter,
    StandardUnetVariantTopology,
)


class _Diffusion(nn.Module):
    """Return one constant complete prediction and count graph calls."""

    def __init__(self, value: float) -> None:
        """Retain the output value."""

        super().__init__()
        self.value = value
        self.calls = 0
        self.contexts: list[torch.Tensor] = []
        self.transformer_options: list[dict[str, object]] = []

    def _forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Return a constant tensor matching the model input."""

        del kwargs
        self.calls += 1
        self.contexts.append(cast(torch.Tensor, args[2]))
        self.transformer_options.append(cast(dict[str, object], args[5]))
        return torch.full_like(cast(torch.Tensor, args[0]), self.value)


class _AttentionResolver:
    """Provide the structural resolver boundary without executing attention."""

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, object],
    ) -> UnetAttn2Execution:
        """Reject callback execution in complete-graph selection tests."""

        del query, context, extra_options
        raise AssertionError("Fake diffusion must not execute attn2 callbacks.")


class _Resolver:
    """Return predeclared resident region shells."""

    def __init__(self, variants: dict[int, _Diffusion]) -> None:
        """Retain region-to-shell ownership."""

        self.variants = variants
        self.cleared = False

    def resolve(
        self,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> nn.Module:
        """Return the selected shell."""

        del schedule_multipliers
        return self.variants[variant.region_index]

    def clear(self) -> None:
        """Record lifecycle cleanup."""

        self.cleared = True


def test_exhaustive_two_region_call_uses_exactly_two_variant_graphs() -> None:
    """Use two exact regional graphs without a hidden base call."""

    base = _Diffusion(0.0)
    left = _Diffusion(1.0)
    right = _Diffusion(3.0)
    resolver = _Resolver({0: left, 1: right})
    execution, phase, context_session, contexts = _execution(base, resolver)
    args, kwargs = _args()

    with (
        phase.activate(kwargs["transformer_options"]),  # type: ignore[arg-type]
        context_session.activate(contexts),
    ):
        result = execution.execute(*args, **kwargs)

    assert base.calls == 0
    assert left.calls == 1
    assert right.calls == 1
    assert left.contexts == [contexts.regions[0].entries[0].context]
    assert right.contexts == [contexts.regions[1].entries[0].context]
    assert left.transformer_options[0] == kwargs["transformer_options"]
    assert right.transformer_options[0] == kwargs["transformer_options"]
    assert "patches" not in left.transformer_options[0]
    assert "patches" not in right.transformer_options[0]
    assert torch.equal(result[0, 0, 0], torch.tensor([1.0, 3.0]))


def test_exhaustive_inactive_region_uses_phased_native_regional_lane() -> None:
    """Use exact regional inputs beside the packed global base lane."""

    base = _Diffusion(5.0)
    left = _Diffusion(1.0)
    resolver = _Resolver({0: left})
    execution, phase, context_session, contexts = _execution(
        base,
        resolver,
        active_regions=(0,),
    )
    args, kwargs = _args()

    with (
        phase.activate(kwargs["transformer_options"]),  # type: ignore[arg-type]
        context_session.activate(contexts),
    ):
        result = execution.execute(*args, **kwargs)

    assert base.contexts == [contexts.regions[1].entries[0].context]
    assert left.contexts == [contexts.regions[0].entries[0].context]
    assert base.transformer_options[0] == kwargs["transformer_options"]
    assert "patches" not in base.transformer_options[0]
    assert left.transformer_options[0] == kwargs["transformer_options"]
    assert "patches" not in left.transformer_options[0]
    assert torch.equal(result[0, 0, 0], torch.tensor([1.0, 5.0]))


def test_genuine_canvas_gap_retains_global_attention_base_graph() -> None:
    """Keep the corrected global base when no authored region owns a pixel."""

    base = _Diffusion(5.0)
    left = _Diffusion(1.0)
    resolver = _Resolver({0: left})
    masks = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
    execution, phase, context_session, contexts = _execution(
        base,
        resolver,
        active_regions=(0,),
        masks=masks,
    )
    args, kwargs = _args(width=3)

    with (
        phase.activate(kwargs["transformer_options"]),  # type: ignore[arg-type]
        context_session.activate(contexts),
    ):
        result = execution.execute(*args, **kwargs)

    assert base.contexts == [contexts.base_context]
    assert left.contexts == [contexts.regions[0].entries[0].context]
    base_patches = base.transformer_options[0].get("patches")
    assert isinstance(base_patches, dict)
    assert len(base_patches["attn2_patch"]) == 1
    assert len(base_patches["attn2_output_patch"]) == 1
    assert left.transformer_options[0] == kwargs["transformer_options"]
    assert "patches" not in left.transformer_options[0]
    assert torch.equal(result[0, 0, 0], torch.tensor([1.0, 5.0, 5.0]))


def _execution(
    base: _Diffusion,
    resolver: _Resolver,
    *,
    active_regions: tuple[int, ...] = (0, 1),
    masks: torch.Tensor | None = None,
) -> tuple[
    StandardUnetVariantExecution,
    StandardUnetAttentionPhaseSession,
    RegionalAttentionExecutionContext,
    BatchedRegionalAttentionContexts,
]:
    """Return one two-region execution with shared phase state."""

    plan = RegionalLoraPlan(
        tuple(
            _plan(index, region, branch)
            for index, (region, branch) in enumerate(
                (
                    (0, RegionalLoraBranch.POSITIVE),
                    (1, RegionalLoraBranch.POSITIVE),
                    (0, RegionalLoraBranch.NEGATIVE),
                    (1, RegionalLoraBranch.NEGATIVE),
                )
            )
        )
    )
    topology = StandardUnetVariantTopology(
        tuple(
            StandardUnetRegionalVariant(
                region_index,
                (_variant_adapter(region_index, region_index),),
            )
            for region_index in active_regions
        )
    )
    if masks is None:
        masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    bank = RegionalMaskBank(
        masks.clone(),
        masks.clone(),
        int(masks.shape[-1]),
        int(masks.shape[-2]),
    )
    phase = StandardUnetAttentionPhaseSession()
    context_session = RegionalAttentionExecutionContext()
    contexts = _contexts()
    return (
        StandardUnetVariantExecution(
            base_diffusion=base,
            plan=plan,
            topology=topology,
            mask_bank=bank,
            variant_resolver=resolver,
            conditioning=StandardUnetVariantConditioningResolver(context_session),
            attention_phase=phase,
            base_attention=StandardUnetVariantBaseAttention(_AttentionResolver()),
        ),
        phase,
        context_session,
        contexts,
    )


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return distinct global, left, and right native context tensors."""

    return BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        torch.zeros((1, 1, 1)),
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, torch.ones((1, 1, 1)), (1.0,)),),
            ),
            BatchedRegionalAttentionRegion(
                1,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.full((1, 1, 1), 2.0),
                        (1.0,),
                    ),
                ),
            ),
        ),
    )


def _plan(
    index: int,
    region: int,
    branch: RegionalLoraBranch,
) -> RegionalLoraAdapterPlan:
    """Return one time-invariant adapter plan."""

    return RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(f"adapter-{region}"),
        index,
        region,
        branch,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )


def _variant_adapter(index: int, region: int) -> StandardUnetVariantAdapter:
    """Return one execution-only branch-neutral adapter."""

    return StandardUnetVariantAdapter(
        index,
        region,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
        (),
    )


def _args(*, width: int = 2) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return Comfy's installed standard-UNet host invocation."""

    options: dict[str, object] = {
        "sample_sigmas": torch.tensor([2.0, 1.0, 0.0]),
        "sigmas": torch.tensor([2.0]),
    }
    return (
        (torch.zeros((1, 4, 1, width)), torch.ones(1)),
        {
            "context": torch.zeros((1, 1, 1)),
            "control": None,
            "transformer_options": options,
        },
    )
