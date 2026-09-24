# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify recoverable completed-call validation for standard UNet."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.runtime.attention_coupling import (
    standard_unet_model_output_validation as output_validation,
)


def test_validator_returns_the_exact_finite_model_output() -> None:
    """Preserve output identity after one completed-call finite observation."""

    model_input = torch.zeros((2, 4, 8, 8))
    output = torch.ones_like(model_input)

    observed = output_validation.StandardUnetModelOutputValidator().validate(
        output,
        model_input=model_input,
    )

    assert observed is output


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (cast(Any, "invalid"), "must return a tensor"),
        (torch.ones((1, 4, 8, 8), dtype=torch.int64), "floating point"),
        (torch.full((1, 4, 8, 8), float("nan")), "non-finite"),
    ],
)
def test_validator_rejects_invalid_completed_model_output(
    output: object,
    message: str,
) -> None:
    """Fail recoverably after the completed denoiser result is available."""

    with pytest.raises((TypeError, ValueError), match=message):
        output_validation.StandardUnetModelOutputValidator().validate(
            output,
            model_input=torch.zeros((1, 4, 8, 8)),
        )
