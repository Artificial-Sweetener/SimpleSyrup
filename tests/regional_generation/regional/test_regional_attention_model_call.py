# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one-call regional CFG, UUID, schedule, and batch resolution."""

from __future__ import annotations

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
from simple_syrup.runtime.regional_attention_model_call import (
    RegionalAttentionModelCallResolver,
)


def test_model_call_resolver_aligns_actual_cfg_order_once_per_diffusion_call() -> None:
    """Resolve positive and negative chunks from exact current base contexts."""

    plan = _plan()
    positive = plan.positive.base_context.entries[0].cross_attention
    negative = plan.negative.base_context.entries[0].cross_attention

    contexts = RegionalAttentionModelCallResolver().resolve(
        plan,
        model_input=torch.zeros(4, 4, 2, 2),
        base_context=torch.cat(
            (negative.expand(2, -1, -1), positive.expand(2, -1, -1))
        ),
        transformer_options={
            "cond_or_uncond": [1, 0],
            "sigmas": torch.tensor([0.5, 0.5]),
        },
    )

    assert contexts.latent_batch_size == 2
    assert [chunk.branch for chunk in contexts.chunks] == [
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
    ]
    assert contexts.base_context[:, 0, 0].tolist() == [3.0, 3.0, 1.0, 1.0]
    assert contexts.regions[0].entries[0].context[:, 0, 0].tolist() == [
        4.0,
        4.0,
        2.0,
        2.0,
    ]


def test_model_call_resolver_uses_runtime_base_device_and_dtype_authority() -> None:
    """Align regional tensors to Comfy's model-consumed context without mutation."""

    plan = _plan()
    source_region = plan.positive.regional_contexts[0].entries[0].cross_attention
    runtime_base = plan.positive.base_context.entries[0].cross_attention.to(
        dtype=torch.float16
    )

    contexts = RegionalAttentionModelCallResolver().resolve(
        plan,
        model_input=torch.zeros(1, 4, 2, 2, dtype=torch.float16),
        base_context=runtime_base,
        transformer_options={
            "cond_or_uncond": [0],
            "sigmas": torch.tensor([0.5]),
        },
    )

    assert contexts.base_context is runtime_base
    assert contexts.regions[0].entries[0].context.dtype is torch.float16
    assert contexts.regions[0].entries[0].context.device == runtime_base.device
    assert source_region.dtype is torch.float32


@pytest.mark.parametrize(
    ("model_input", "options", "message"),
    [
        (
            torch.zeros(3, 4, 2, 2),
            {"cond_or_uncond": [0, 1], "sigmas": torch.tensor([0.5])},
            "divide evenly",
        ),
        (
            torch.zeros(1, 4, 2, 2),
            {"cond_or_uncond": [0], "sigmas": torch.tensor([0.5, 0.25])},
            "uniform",
        ),
        (
            torch.zeros(1, 4, 2, 2),
            {"cond_or_uncond": [], "sigmas": torch.tensor([0.5])},
            "require cond_or_uncond",
        ),
    ],
)
def test_model_call_resolver_rejects_invalid_call_geometry_before_alignment(
    model_input: torch.Tensor,
    options: dict[str, object],
    message: str,
) -> None:
    """Fail closed on malformed CFG and sigma call metadata."""

    plan = _plan()

    with pytest.raises((TypeError, ValueError), match=message):
        RegionalAttentionModelCallResolver().resolve(
            plan,
            model_input=model_input,
            base_context=plan.positive.base_context.entries[0].cross_attention,
            transformer_options=options,
        )


def _plan() -> ProcessedRegionalAttentionPlan:
    """Return one always-active positive/negative plan."""

    masks = torch.ones(1, 1, 1)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0), (_context(1, 0, 2.0),)
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, 3.0), (_context(1, 0, 4.0),)
        ),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one exact processed context with stable schedule identity."""

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
