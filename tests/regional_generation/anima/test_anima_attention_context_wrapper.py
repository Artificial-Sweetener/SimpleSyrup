# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify dynamic CFG and schedule context publication for patched Anima."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.tiled_diffusion import LatentTile
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
from simple_syrup.runtime.spatial_model_arguments import make_tiled_model_args


class _DiffusionModel(torch.nn.Module):
    """Provide weak-referenceable model identity for wrapper tests."""


@dataclass
class _Executor:
    """Expose Comfy's diffusion wrapper executor surface."""

    class_obj: object
    execution: AnimaRegionalAttentionExecution
    observed: object = None

    def __call__(self, *_: object, **__: object) -> torch.Tensor:
        """Capture active contexts and return one model-shaped tensor."""

        self.observed = self.execution.active_contexts
        return torch.ones((2, 1, 1, 1, 1))


def test_wrapper_publishes_current_cfg_chunks_and_restores_template() -> None:
    """Align both CFG branches by active base context for one exact call."""

    plan = _processed_plan()
    template = build_regional_attention_template(plan, latent_batch_size=1)
    execution = AnimaRegionalAttentionExecution(
        template,
        plan.mask_bank,
        (1.0,),
        dynamic_contexts=True,
    )
    diffusion_model = _DiffusionModel()
    surface = AnimaModuleSurface(cast(Any, diffusion_model), (), ())
    wrapper = AnimaRegionalAttentionContextDiffusionWrapper(
        surface,
        plan,
        execution,
    )
    executor = _Executor(diffusion_model, execution)
    positive = plan.positive.base_context.entries[0].cross_attention
    negative = plan.negative.base_context.entries[0].cross_attention

    output = wrapper(
        executor,
        torch.zeros((2, 16, 1, 2, 2)),
        torch.tensor([0.5, 0.5]),
        torch.cat((positive, negative)),
        None,
        None,
        transformer_options={
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.5, 0.5]),
        },
    )

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (2, 1, 1, 1, 1)
    observed = cast(Any, executor.observed)
    assert [chunk.branch for chunk in observed.chunks] == [
        RegionalAttentionBranch.POSITIVE,
        RegionalAttentionBranch.NEGATIVE,
    ]
    assert torch.equal(observed.base_context, torch.cat((positive, negative)))
    assert execution.active_contexts is template


def test_wrapper_uses_base_context_for_an_absent_regional_prompt() -> None:
    """Keep a technical zero-mask region valid for global-only conditioning."""

    plan = _global_only_processed_plan()
    template = build_regional_attention_template(plan, latent_batch_size=1)
    template_entry = template.regions[0].entries[0]
    positive = plan.positive.base_context.entries[0].cross_attention
    negative = plan.negative.base_context.entries[0].cross_attention

    assert torch.equal(template_entry.context, positive)
    assert template_entry.strengths == (1.0,)

    execution = AnimaRegionalAttentionExecution(
        template,
        plan.mask_bank,
        (1.0,),
        dynamic_contexts=True,
    )
    diffusion_model = _DiffusionModel()
    wrapper = AnimaRegionalAttentionContextDiffusionWrapper(
        AnimaModuleSurface(cast(Any, diffusion_model), (), ()),
        plan,
        execution,
    )
    executor = _Executor(diffusion_model, execution)

    wrapper(
        executor,
        torch.zeros((2, 16, 1, 2, 2)),
        torch.tensor([0.5, 0.5]),
        torch.cat((positive, negative)),
        None,
        None,
        transformer_options={
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.5, 0.5]),
        },
    )

    observed = cast(Any, executor.observed)
    observed_entry = observed.regions[0].entries[0]
    assert torch.equal(observed_entry.context, torch.cat((positive, negative)))
    assert observed_entry.strengths == (1.0, 1.0)
    assert execution.active_contexts is template


@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "selectors",
    [(0,), (0, 1)],
    ids=("positive-only", "ordinary-cfg"),
)
def test_wrapper_aligns_view_repeated_cfg_chunks_over_latent_batches(
    tile_batch_size: int,
    selectors: tuple[int, ...],
) -> None:
    """Preserve view, CFG chunk, then latent-sample order for tiled Anima."""

    plan = _processed_plan()
    template = build_regional_attention_template(plan, latent_batch_size=2)
    execution = AnimaRegionalAttentionExecution(
        template,
        plan.mask_bank,
        (1.0,),
        dynamic_contexts=True,
    )
    diffusion_model = _DiffusionModel()
    wrapper = AnimaRegionalAttentionContextDiffusionWrapper(
        AnimaModuleSurface(cast(Any, diffusion_model), (), ()),
        plan,
        execution,
    )
    executor = _Executor(diffusion_model, execution)
    positive = plan.positive.base_context.entries[0].cross_attention
    negative = plan.negative.base_context.entries[0].cross_attention
    base_context = torch.cat(
        tuple(
            (positive if selector == 0 else negative).expand(2, -1, -1)
            for selector in selectors
        )
    )
    source_batch_size = len(selectors) * 2
    canvas_width = tile_batch_size * 2
    source_args: dict[str, Any] = {
        "input": torch.zeros((source_batch_size, 16, 1, 2, canvas_width)),
        "timestep": torch.full((source_batch_size,), 0.5),
        "cond_or_uncond": list(selectors),
        "c": {
            "cross_attn": base_context,
            "transformer_options": {
                "cond_or_uncond": list(selectors),
                "uuids": tuple(
                    (
                        plan.positive.base_context.entries[0].uuid
                        if selector == 0
                        else plan.negative.base_context.entries[0].uuid
                    )
                    for selector in selectors
                ),
                "sigmas": torch.full((2,), 0.5),
            },
        },
    }
    tiled = make_tiled_model_args(
        args=source_args,
        tiles=tuple(LatentTile(index * 2, 0, 2, 2) for index in range(tile_batch_size)),
        input_batch_size=source_batch_size,
        latent_height=2,
        latent_width=canvas_width,
    )
    conditioning = cast(dict[str, Any], tiled["c"])
    options = cast(dict[str, Any], conditioning["transformer_options"])
    tiled_context = cast(torch.Tensor, conditioning["cross_attn"])

    wrapper(
        executor,
        cast(torch.Tensor, tiled["input"]),
        cast(torch.Tensor, tiled["timestep"]),
        tiled_context,
        None,
        None,
        transformer_options=options,
    )

    observed = cast(Any, executor.observed)
    assert observed.latent_batch_size == 2
    assert [chunk.branch for chunk in observed.chunks] == [
        (
            RegionalAttentionBranch.POSITIVE
            if selector == 0
            else RegionalAttentionBranch.NEGATIVE
        )
        for _view_index in range(tile_batch_size)
        for selector in selectors
    ]
    assert observed.base_context[:, 0, 0].tolist() == [
        value
        for _view_index in range(tile_batch_size)
        for selector in selectors
        for value in ((1.0, 1.0) if selector == 0 else (3.0, 3.0))
    ]
    assert options["cond_or_uncond"] == list(selectors) * tile_batch_size
    assert execution.active_contexts is template


def test_wrapper_requires_installed_anima_positional_context_contract() -> None:
    """Reject calls that omit the third positional Anima context argument."""

    plan = _processed_plan()
    template = build_regional_attention_template(plan, latent_batch_size=1)
    execution = AnimaRegionalAttentionExecution(
        template,
        plan.mask_bank,
        (1.0,),
        dynamic_contexts=True,
    )
    diffusion_model = _DiffusionModel()
    wrapper = AnimaRegionalAttentionContextDiffusionWrapper(
        AnimaModuleSurface(cast(Any, diffusion_model), (), ()),
        plan,
        execution,
    )

    with pytest.raises(TypeError, match="third positional"):
        wrapper(
            _Executor(diffusion_model, execution),
            torch.zeros((1, 16, 1, 2, 2)),
            torch.tensor([0.5]),
            transformer_options={
                "cond_or_uncond": [0],
                "sigmas": torch.tensor([0.5]),
            },
        )


def _processed_plan() -> ProcessedRegionalAttentionPlan:
    """Return one positive/negative plan with distinct recognizable tensors."""

    positive_base = _context(0, None, 1.0)
    positive_region = _context(1, 0, 2.0)
    negative_base = _context(0, None, 3.0)
    negative_region = _context(1, 0, 4.0)
    masks = torch.ones((1, 1, 1))
    bank = RegionalMaskBank(masks, masks.clone(), 1, 1)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(positive_base, (positive_region,)),
        ProcessedRegionalAttentionBranch(negative_base, (negative_region,)),
        bank,
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _global_only_processed_plan() -> ProcessedRegionalAttentionPlan:
    """Return one technical mask bank with no authored regional contexts."""

    positive_base = _context(0, None, 1.0)
    negative_base = _context(0, None, 3.0)
    masks = torch.zeros((1, 1, 1))
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(positive_base, ()),
        ProcessedRegionalAttentionBranch(negative_base, ()),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active processed conditioning context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 2, 3), value),
                1.0,
            ),
        ),
    )
