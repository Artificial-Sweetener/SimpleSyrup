# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only live NegPiP callback introspection."""

from __future__ import annotations

import torch

from simple_syrup.runtime.negpip.krea2 import TRANSFORMER_MASK_KEY
from tools.attention_coupling_benchmark.comfy_probe.negpip_runtime import (
    InstrumentNegpipModelV3,
    ReadNegpipRuntimeV3,
    _NegpipProbeState,
    _observe_masked,
    _observe_standard,
    _owned_negpip_callback,
)


def test_negpip_runtime_probe_schemas_are_stable() -> None:
    """Expose distinct instrument and synchronized evidence node IDs."""

    instrument = InstrumentNegpipModelV3.define_schema()
    reader = ReadNegpipRuntimeV3.define_schema()

    assert instrument.node_id == "SimpleSyrupBenchmark.InstrumentNegpipModel"
    assert reader.node_id == "SimpleSyrupBenchmark.ReadNegpipRuntime"


def test_standard_probe_validates_live_interleaved_selection() -> None:
    """Record one standard callback only after exact even/odd selection."""

    state = _NegpipProbeState("standard", "attn2_patch", "callback")
    query = torch.ones((1, 2, 1))
    key = torch.tensor([[[1.0], [1.0], [2.0], [2.0]]])
    value = torch.tensor([[[1.0], [-1.0], [2.0], [2.0]]])

    _observe_standard(
        state,
        query,
        key,
        value,
        (query, key[:, 0::2], value[:, 1::2]),
    )

    assert state.attention_calls == 1
    assert state.negative_mask_calls == 1
    assert state.input_value_shape == [1, 4, 1]
    assert state.output_value_shape == [1, 2, 1]
    assert state.negative_token_count == 1
    assert state.negative_token_positions == [0]
    assert state.negative_token_locations == [[0, 0]]


def test_standard_probe_finds_signed_tokens_outside_cfg_batch_zero() -> None:
    """Inspect every CFG row rather than assuming the positive prompt is first."""

    state = _NegpipProbeState("standard", "attn2_patch", "callback")
    query = torch.ones((2, 2, 1))
    key = torch.tensor(
        [
            [[1.0], [1.0], [2.0], [2.0]],
            [[1.0], [1.0], [2.0], [2.0]],
        ]
    )
    value = torch.tensor(
        [
            [[1.0], [1.0], [2.0], [2.0]],
            [[1.0], [1.0], [2.0], [-2.0]],
        ]
    )

    _observe_standard(
        state,
        query,
        key,
        value,
        (query, key[:, 0::2], value[:, 1::2]),
    )

    assert state.negative_mask_calls == 1
    assert state.negative_token_count == 1
    assert state.negative_token_positions == [1]
    assert state.negative_token_locations == [[1, 1]]


def test_probe_admits_the_pinned_ppm_standard_callback_identity() -> None:
    """Recognize PPM even when Comfy prefixes its module with a Windows path."""

    def sdxl_attn2_negpip() -> None:
        """Stand in for the identity-checked pinned PPM callback."""

    sdxl_attn2_negpip.__module__ = "managed_comfyui_ppm.src.negpip.unet_negpip"
    sdxl_attn2_negpip.__qualname__ = "sdxl_attn2_negpip"

    patch_name, family, callback = _owned_negpip_callback(
        {"attn2_patch": [sdxl_attn2_negpip]}
    )

    assert patch_name == "attn2_patch"
    assert family == "standard"
    assert callback is sdxl_attn2_negpip


def test_krea_probe_validates_text_only_value_signing() -> None:
    """Record Krea only when its image suffix remains exact."""

    state = _NegpipProbeState("krea2", "attn1_patch", "callback")
    query = torch.ones((1, 1, 4, 1))
    key = torch.ones((1, 1, 4, 1)) * 2
    value = torch.tensor([[[[3.0], [4.0], [5.0], [6.0]]]])
    multiplier = torch.tensor([[[-1.0], [1.0]]])
    output_value = value.clone()
    output_value[:, :, :2] *= multiplier.unsqueeze(1)

    _observe_masked(
        state,
        query,
        key,
        value,
        {"q": query, "k": key, "v": output_value},
        {TRANSFORMER_MASK_KEY: multiplier, "img_slice": [2, 4]},
    )

    assert state.attention_calls == 1
    assert state.negative_mask_calls == 1
    assert state.text_length == 2
    assert state.mask_shape == [1, 2, 1]
    assert state.negative_token_count == 1
    assert state.negative_token_positions == [0]
    assert state.negative_token_locations == [[0, 0]]
