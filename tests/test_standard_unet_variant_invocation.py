# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Comfy host-call normalization for persistent UNet variants."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_lora.standard_unet_variant_invocation import (
    StandardUnetVariantInvocation,
)


def test_keyword_host_call_normalizes_wrapper_arguments() -> None:
    """Bind Comfy's two-positional plus keyword diffusion call exactly once."""

    model_input = torch.zeros((2, 4, 8, 8))
    timesteps = torch.ones(2)
    context = torch.zeros((2, 1, 4))
    control = {"input": [torch.ones(1)]}
    options: dict[str, object] = {"sample_sigmas": torch.tensor([1.0, 0.0])}

    invocation = StandardUnetVariantInvocation.bind(
        (model_input, timesteps),
        {
            "context": context,
            "control": control,
            "transformer_options": options,
            "extra": "preserved",
        },
    )
    graph_args, graph_kwargs = invocation.graph_call(
        context=context,
        transformer_options=options,
    )

    assert graph_args[:5] == (model_input, timesteps, context, None, control)
    assert graph_args[5] == options
    assert graph_args[5] is not options
    assert graph_args[4] is not control
    assert graph_args[4]["input"] is not control["input"]  # type: ignore[index]
    assert graph_kwargs == {"extra": "preserved"}


def test_graph_calls_isolate_mutable_host_metadata() -> None:
    """Give every complete graph independent transformer and control state."""

    options: dict[str, object] = {"transformer_index": 0}
    control = {"output": [torch.ones(1)]}
    invocation = StandardUnetVariantInvocation.bind(
        (torch.zeros((1, 4, 1, 1)), torch.ones(1)),
        {
            "context": torch.zeros((1, 1, 1)),
            "control": control,
            "transformer_options": options,
        },
    )

    context = torch.ones((1, 1, 1))
    first_args, _ = invocation.graph_call(
        context=context,
        transformer_options=options,
    )
    second_args, _ = invocation.graph_call(
        context=context,
        transformer_options=options,
    )
    first_options = first_args[5]
    first_control = first_args[4]
    assert isinstance(first_options, dict)
    assert isinstance(first_control, dict)
    first_options["transformer_index"] = 9
    first_control["output"].pop()

    assert second_args[5] == {"transformer_index": 0}
    assert len(second_args[4]["output"]) == 1  # type: ignore[index]
    assert options == {"transformer_index": 0}
    assert len(control["output"]) == 1


def test_graph_call_installs_the_exact_native_context() -> None:
    """Replace only the graph context while preserving the source invocation."""

    source = torch.zeros((1, 2, 3))
    regional = torch.ones((1, 2, 3))
    invocation = StandardUnetVariantInvocation.bind(
        (torch.zeros((1, 4, 1, 1)), torch.ones(1)),
        {"context": source, "transformer_options": {}},
    )

    prepared_options: dict[str, object] = {"patches": {"attn2_patch": [object()]}}
    arguments, _ = invocation.graph_call(
        context=regional,
        transformer_options=prepared_options,
    )

    assert arguments[2] is regional
    assert arguments[5] == prepared_options
    assert arguments[5] is not prepared_options
    assert invocation.arguments[2] is source
