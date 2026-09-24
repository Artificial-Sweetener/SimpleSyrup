# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify typed PPM NegPiP conditioning and call-local option adaptation."""

from __future__ import annotations

from types import SimpleNamespace

import comfy.conds
import pytest
import torch

from simple_syrup.runtime.ppm_negpip_interop import (
    PpmNegpipInterop,
    PpmNegpipSemantics,
)


def test_anima_adapter_extracts_exact_typed_value_multiplier() -> None:
    """Convert PPM's model condition to context-aligned execution state."""

    interop = _anima_interop()
    context = torch.zeros((1, 3, 4), dtype=torch.float16)
    source = torch.tensor([[[1], [-1], [1]]], dtype=torch.int32)

    multiplier = interop.extract_value_multiplier(
        {"c_ppm_negpip_mask": comfy.conds.CONDRegular(source)},
        context,
    )

    assert multiplier is not None
    assert multiplier.dtype is context.dtype
    assert multiplier.device == context.device
    assert multiplier.tolist() == [[[1.0], [-1.0], [1.0]]]


def test_anima_adapter_uses_neutral_multiplier_when_condition_is_absent() -> None:
    """Represent an all-positive prompt without leaving branch state partial."""

    context = torch.zeros((2, 3, 4))

    multiplier = _anima_interop().extract_value_multiplier({}, context)

    assert multiplier is not None
    assert torch.equal(multiplier, torch.ones((2, 3, 1)))


@pytest.mark.parametrize(
    "source",
    [
        torch.ones((1, 2, 1)),
        torch.zeros((1, 3, 1)),
        torch.ones((1, 3, 2)),
    ],
)
def test_anima_adapter_rejects_misaligned_or_nonbinary_masks(
    source: torch.Tensor,
) -> None:
    """Fail closed before malformed PPM state reaches regional packing."""

    with pytest.raises(ValueError):
        _anima_interop().extract_value_multiplier(
            {"c_ppm_negpip_mask": SimpleNamespace(cond=source)},
            torch.zeros((1, 3, 4)),
        )


def test_anima_adapter_publishes_mask_on_an_isolated_option_copy() -> None:
    """Keep the ordinary call options unchanged outside original cross-attention."""

    old_mask = torch.ones((1, 2, 1))
    packed = torch.tensor([[[1.0], [-1.0]], [[-1.0], [1.0]]])
    source: dict[str, object] = {
        "ppm_negpip_mask": old_mask,
        "preserved": object(),
    }

    prepared = _anima_interop().prepare_anima_transformer_options(source, packed)

    assert prepared is not source
    assert prepared["preserved"] is source["preserved"]
    assert prepared["ppm_negpip_mask"] is packed
    assert source["ppm_negpip_mask"] is old_mask


def _anima_interop() -> PpmNegpipInterop:
    """Return one focused admitted Anima semantic adapter."""

    return PpmNegpipInterop(
        PpmNegpipSemantics.ANIMA_VALUE_MASK,
        lambda *args, **kwargs: (args, kwargs),
    )
