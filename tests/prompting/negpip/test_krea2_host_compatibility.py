# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safeguard Krea NegPiP at both supported Comfy model call boundaries."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from comfy.ldm.krea2.model import Attention, SingleStreamDiT
from comfy.model_patcher import ModelPatcher

from simple_syrup.runtime.negpip.krea2 import (
    CONDITION_MASK_KEY,
    TRANSFORMER_MASK_KEY,
    krea2_attn1_negpip,
    krea2_diffusion_negpip_wrapper,
)
from simple_syrup.runtime.negpip.krea2_host import (
    Krea2HostAttention,
    krea2_host_mutations,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE


@pytest.mark.parametrize("reference_argument", (False, True))
def test_diffusion_wrapper_preserves_both_host_signatures(
    reference_argument: bool,
) -> None:
    """Inject one local mask without binding transformer options twice."""
    source_options: dict[str, object] = {"existing": True}
    mask = torch.tensor([[[-1.0], [1.0]]])

    def earlier(
        x: object,
        timesteps: object,
        context: object,
        attention_mask: object,
        transformer_options: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        """Expose Comfy 0.28's exact five-positional model boundary."""
        del x, timesteps, context, attention_mask, kwargs
        return transformer_options

    def current(
        x: object,
        timesteps: object,
        context: object,
        attention_mask: object,
        ref_latents: object,
        transformer_options: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        """Expose Comfy's reference-capable six-positional model boundary."""
        del x, timesteps, context, attention_mask, kwargs
        assert ref_latents is None
        return transformer_options

    args = (
        (None, None, None, None, None, source_options)
        if reference_argument
        else (None, None, None, None, source_options)
    )
    prepared = cast(
        dict[str, object],
        krea2_diffusion_negpip_wrapper(
            current if reference_argument else earlier,
            *args,
            **{CONDITION_MASK_KEY: mask},
        ),
    )
    assert prepared is not source_options
    assert prepared[TRANSFORMER_MASK_KEY] is mask
    assert prepared["existing"] is True
    assert TRANSFORMER_MASK_KEY not in source_options


@pytest.mark.parametrize("negative_sign", (False, True))
def test_host_attention_matches_native_math_and_keeps_call_options_local(
    negative_sign: bool,
) -> None:
    """Earlier host hooks reproduce native grouped attention without source mutation."""
    torch.manual_seed(4)
    attention = Attention(128, 4, kvheads=2, operations=torch.nn)
    with torch.no_grad():
        attention.qknorm.qnorm.scale.zero_()
        attention.qknorm.knorm.scale.zero_()
    x = torch.randn(1, 7, 128)
    mask = torch.tensor([[[-1.0 if negative_sign else 1.0], [1.0]]])
    options = {
        TRANSFORMER_MASK_KEY: mask,
        "patches": {"attn1_patch": [krea2_attn1_negpip]},
    }
    native_options = {**options, "block_index": 0, "img_slice": [2, 7]}
    with torch.no_grad():
        expected = attention(x, transformer_options=native_options)
        actual = Krea2HostAttention(attention, 0, 28)(x, transformer_options=options)
    assert torch.equal(actual, expected)
    assert "block_index" not in options
    assert "img_slice" not in options


def test_host_attention_rejects_malformed_token_mask() -> None:
    """Malformed sign metadata must fail before executing attention projections."""
    with pytest.raises(ValueError, match="processed token sign mask"):
        Krea2HostAttention(None, 0, 28)(
            torch.zeros(1, 3, 4), transformer_options={TRANSFORMER_MASK_KEY: "invalid"}
        )


def test_host_attention_delegates_unsigned_calls() -> None:
    """Unmarked conditioning retains the host's ordinary attention operation."""
    torch.manual_seed(4)
    attention = Attention(128, 4, kvheads=2, operations=torch.nn)
    with torch.no_grad():
        attention.qknorm.qnorm.scale.zero_()
        attention.qknorm.knorm.scale.zero_()
        x = torch.randn(1, 7, 128)
        expected = attention(x)
        actual = Krea2HostAttention(attention, 0, 28)(x)
    assert torch.equal(actual, expected)


class _EarlierModel(torch.nn.Module):
    """Expose the earlier host boundary around one real Comfy attention module."""

    def __init__(self) -> None:
        """Retain a genuine patchable module hierarchy without loading weights."""
        super().__init__()
        block = torch.nn.Module()
        block.attn = Attention(128, 4, kvheads=2, operations=torch.nn)
        self.blocks = torch.nn.ModuleList([block])

    def _forward(
        self,
        x: object,
        timesteps: object,
        context: object,
        attention_mask: object = None,
        transformer_options: object = None,
        **kwargs: object,
    ) -> object:
        """Represent only the external host argument contract under test."""
        del timesteps, context, attention_mask, transformer_options, kwargs
        return x


def test_earlier_host_object_patches_restore_on_unload() -> None:
    """Patch only a derived MODEL and restore the shared host method on unload."""
    root = torch.nn.Module()
    root.diffusion_model = _EarlierModel()
    device = torch.device("cpu")
    source = ModelPatcher(root, load_device=device, offload_device=device)
    attention = root.diffusion_model.blocks[0].attn
    assert isinstance(attention, torch.nn.Module)
    original = attention.forward
    derived = PATCHER_LIFECYCLE.derive_model(
        source, krea2_host_mutations(source), operation="test Krea host adaptation"
    )
    assert source.object_patches == {}
    assert attention.forward == original
    derived.patch_model(load_weights=False)
    try:
        assert isinstance(attention.forward, Krea2HostAttention)
    finally:
        derived.unpatch_model(unpatch_weights=False)
    assert attention.forward == original


def test_reference_capable_host_needs_no_object_replacements() -> None:
    """Leave native Krea attention and its reference-image path entirely untouched."""
    root = torch.nn.Module()
    diffusion = object.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(diffusion)
    root.diffusion_model = diffusion
    device = torch.device("cpu")
    model = ModelPatcher(root, load_device=device, offload_device=device)
    assert krea2_host_mutations(model) == ()
