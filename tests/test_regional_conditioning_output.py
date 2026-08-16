# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove Comfy-equivalent within-region active-entry output combination."""

from __future__ import annotations

import pytest
import torch
from comfy.samplers import get_area_and_mult

from simple_syrup.domain.regional_conditioning_output import (
    REGIONAL_CONDITIONING_OUTPUT_COMBINER,
)


@pytest.mark.parametrize(
    ("strengths", "expected"),
    [
        (((1.0,),), 2.0),
        (((0.25,), (0.75,)), 5.0),
        (((-0.5,), (1.5,)), 8.0),
        (((0.0,), (0.0,)), 0.0),
    ],
)
def test_combiner_matches_native_comfy_strength_normalization(
    strengths: tuple[tuple[float, ...], ...],
    expected: float,
) -> None:
    """Use Comfy's weighted sum and `1e-37` denominator initialization."""

    outputs = (torch.tensor([[[2.0]]]), torch.tensor([[[6.0]]]))[: len(strengths)]

    combined = REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
        outputs,
        strengths=strengths,
    )

    assert combined.item() == pytest.approx(expected)


def test_combiner_preserves_independent_batch_strengths() -> None:
    """Combine each latent/chunk sample from its aligned entry strengths."""

    combined = REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
        (
            torch.tensor([[[2.0]], [[10.0]]]),
            torch.tensor([[[6.0]], [[20.0]]]),
        ),
        strengths=((0.25, 1.0), (0.75, 0.0)),
    )

    assert combined[:, 0, 0].tolist() == pytest.approx([5.0, 10.0])


def test_combiner_reuses_one_output_when_every_strength_is_unity() -> None:
    """Avoid numerical work when Comfy normalization is an exact identity."""

    output = torch.randn((2, 4, 8), dtype=torch.float16)

    combined = REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
        (output,),
        strengths=((1.0, 1.0),),
    )

    assert combined is output


def test_combiner_returns_exact_zero_for_inactive_float16_rows() -> None:
    """Avoid denominator underflow when schedule selection prunes a region."""

    combined = REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
        (torch.ones((2, 1, 1), dtype=torch.float16),),
        strengths=((0.0, 1.0),),
    )

    assert torch.isfinite(combined).all()
    assert combined[:, 0, 0].tolist() == [0.0, 1.0]


def test_combiner_matches_installed_comfy_entry_multipliers() -> None:
    """Use installed Comfy's condition multiplier owner as the reference."""

    model_input = torch.zeros((1, 1, 1, 1))
    timestep = torch.ones((1,))
    first = get_area_and_mult(
        {"model_conds": {}, "uuid": "first", "strength": 0.25},
        model_input,
        timestep,
    )
    second = get_area_and_mult(
        {"model_conds": {}, "uuid": "second", "strength": 0.75},
        model_input,
        timestep,
    )
    assert first is not None and second is not None
    first_output = torch.tensor([[[2.0]]])
    second_output = torch.tensor([[[6.0]]])
    expected = (
        first_output * first.mult.flatten()[0]
        + second_output * second.mult.flatten()[0]
    ) / (1e-37 + first.mult.flatten()[0] + second.mult.flatten()[0])

    observed = REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
        (first_output, second_output),
        strengths=((0.25,), (0.75,)),
    )

    torch.testing.assert_close(observed, expected)


def test_combiner_rejects_misaligned_strength_and_output_axes() -> None:
    """Fail before numerical work when active-entry axes disagree."""

    output = torch.ones((1, 2, 3))
    with pytest.raises(ValueError, match="align with ordered outputs"):
        REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
            (output,),
            strengths=((1.0,), (1.0,)),
        )
    with pytest.raises(ValueError, match="strengths must match the output batch"):
        REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
            (output,),
            strengths=((1.0, 1.0),),
        )
