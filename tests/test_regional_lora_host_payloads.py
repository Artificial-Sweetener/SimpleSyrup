"""Characterize installed Comfy WeightHook payload factory boundaries."""

from __future__ import annotations

import comfy.hooks
import torch


def test_raw_lora_factory_retains_payload_identity_and_independent_strengths() -> None:
    """Preserve raw loader mappings until model-family admission."""

    weights = {"fixture.lora_A.weight": torch.ones((1, 2))}
    hooks = comfy.hooks.create_hook_lora(weights, 0.75, -0.25)
    hook = _only_weight_hook(hooks)

    assert hook.weights is weights
    assert hook.need_weight_init is True
    assert hook._strength_model == 0.75
    assert hook._strength_clip == -0.25


def test_model_as_lora_factory_marks_converted_patch_payloads() -> None:
    """Expose the exact converted shape rejected by standard-LoRA decoding."""

    model_weight = torch.ones((2, 2))
    hooks = comfy.hooks.create_hook_model_as_lora(
        weights_model={"fixture.weight": model_weight},
        weights_clip=None,
        strength_model=0.5,
        strength_clip=0.0,
    )
    hook = _only_weight_hook(hooks)

    assert hook.need_weight_init is False
    assert hook.weights == {"fixture.weight": ("model_as_lora", (model_weight,))}
    assert hook.weights["fixture.weight"][1][0] is model_weight


def _only_weight_hook(hooks: comfy.hooks.HookGroup) -> comfy.hooks.WeightHook:
    """Return one installed WeightHook from one factory result."""

    assert len(hooks.hooks) == 1
    hook = hooks.hooks[0]
    assert isinstance(hook, comfy.hooks.WeightHook)
    return hook
