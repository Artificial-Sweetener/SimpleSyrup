# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the fixed regional Attention Coupling MODEL patch-stack order."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from simple_syrup.runtime.model_patcher_mutations import (
    ModelDiffusionWrapperMutation,
    ModelUnetWrapperMutation,
)
from simple_syrup.runtime.patcher_lifecycle import ModelMutation
from simple_syrup.runtime.regional_model_patch_stack import (
    REGIONAL_MODEL_PATCH_STACK_BUILDER,
    RegionalModelPatchStack,
)


def test_regional_patch_stack_preserves_state_lineage_and_runtime_nesting() -> None:
    """Execute upstream, spatial, attention, and inner layers in fixed order."""

    from comfy.patcher_extension import WrapperExecutor

    events: list[str] = []
    source = _patcher(torch.nn.Linear(1, 1))
    global_lora_patch = ("global-lora-patch",)
    source.patches["weight"] = [global_lora_patch]

    def upstream_model_wrapper(
        apply_model: Callable[..., object],
        args: dict[str, object],
    ) -> object:
        """Preserve one user-supplied outer model wrapper."""

        events.append("upstream-model-enter")
        result = apply_model(args["input"], args["timestep"])
        events.append("upstream-model-exit")
        return result

    def upstream_diffusion_wrapper(
        executor: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Preserve one user-supplied diffusion wrapper."""

        events.append("upstream-diffusion-enter")
        result = executor(*args, **kwargs)
        events.append("upstream-diffusion-exit")
        return result

    source.set_model_unet_function_wrapper(upstream_model_wrapper)
    source.add_wrapper_with_key(
        "diffusion_model",
        "upstream.diffusion",
        upstream_diffusion_wrapper,
    )

    def attention_wrapper(
        executor: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Represent the Attention Coupling diffusion boundary."""

        events.append("attention-enter")
        result = executor(*args, **kwargs)
        events.append("attention-exit")
        return result

    def spatial_wrapper(
        apply_model: Callable[..., object],
        args: dict[str, object],
    ) -> object:
        """Represent the outer tiled or Contextual model-function boundary."""

        events.append("spatial-enter")
        result = upstream_model_wrapper(apply_model, args)
        events.append("spatial-exit")
        return result

    stack = REGIONAL_MODEL_PATCH_STACK_BUILDER.build(
        source,
        attention_mutations=(
            ModelDiffusionWrapperMutation(
                "simple_syrup.attention_coupling",
                attention_wrapper,
            ),
        ),
        spatial_mutations=(ModelUnetWrapperMutation(spatial_wrapper),),
    )

    assert stack.user_model is source
    assert stack.attention_model.parent is source
    assert stack.sampling_model.parent is source
    assert stack.sampling_model.patches["weight"] == [global_lora_patch]
    assert (
        source.get_wrappers("diffusion_model", "simple_syrup.attention_coupling") == []
    )
    assert stack.attention_model.get_all_wrappers("diffusion_model") == [
        upstream_diffusion_wrapper,
        attention_wrapper,
    ]
    assert stack.sampling_model.get_all_wrappers("diffusion_model") == [
        upstream_diffusion_wrapper,
        attention_wrapper,
    ]
    assert source.model_options["model_function_wrapper"] is upstream_model_wrapper
    assert (
        stack.attention_model.model_options["model_function_wrapper"]
        is upstream_model_wrapper
    )
    assert (
        stack.sampling_model.model_options["model_function_wrapper"] is spatial_wrapper
    )

    def inner_diffusion(*args: object, **kwargs: object) -> str:
        """Record the innermost diffusion evaluation and preserved LoRA state."""

        del args, kwargs
        assert stack.sampling_model.patches["weight"] == [global_lora_patch]
        events.append("inner-diffusion")
        return "prediction"

    def apply_model(*args: object, **kwargs: object) -> object:
        """Execute the installed Comfy diffusion-wrapper chain."""

        del args, kwargs
        wrappers = stack.sampling_model.get_all_wrappers("diffusion_model")
        return WrapperExecutor.new_executor(inner_diffusion, wrappers).execute()

    result = stack.sampling_model.model_options["model_function_wrapper"](
        apply_model,
        {"input": object(), "timestep": object()},
    )

    assert result == "prediction"
    assert events == [
        "spatial-enter",
        "upstream-model-enter",
        "upstream-diffusion-enter",
        "attention-enter",
        "inner-diffusion",
        "attention-exit",
        "upstream-diffusion-exit",
        "upstream-model-exit",
        "spatial-exit",
    ]


@pytest.mark.parametrize(
    ("attention_mutations", "spatial_mutations", "message"),
    [
        ((), (ModelUnetWrapperMutation(lambda *args: object()),), "Attention Coupling"),
        (
            (ModelDiffusionWrapperMutation("simple_syrup.attention", lambda: None),),
            (),
            "spatial mutations",
        ),
    ],
)
def test_regional_patch_stack_requires_both_derivation_stages(
    attention_mutations: tuple[ModelMutation, ...],
    spatial_mutations: tuple[ModelMutation, ...],
    message: str,
) -> None:
    """Reject an incomplete stack before cloning the user MODEL."""

    source = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match=message):
        REGIONAL_MODEL_PATCH_STACK_BUILDER.build(
            source,
            attention_mutations=attention_mutations,
            spatial_mutations=spatial_mutations,
        )

    assert source.parent is None


def test_regional_patch_stack_value_is_immutable() -> None:
    """Prevent callers from rewriting recorded MODEL generation ownership."""

    stack = RegionalModelPatchStack(object(), object(), object())

    with pytest.raises(FrozenInstanceError):
        stack.sampling_model = object()  # type: ignore[misc]


def _patcher(model: torch.nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
