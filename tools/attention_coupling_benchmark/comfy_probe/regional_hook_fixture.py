# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Construct exact rejection-only Comfy hook fixtures for managed evidence."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import comfy.hooks
import torch

NON_WEIGHT_HOOK = "non-weight-hook"
MODEL_AS_LORA_FORMAT = "model-as-lora-format"
UNSUPPORTED_SUFFIX_FORMAT = "unsupported-suffix-format"
INCOMPLETE_STANDARD_PAIR = "incomplete-standard-pair"
UNSUPPORTED_ANIMA_TARGET = "unsupported-anima-target"
LLM_ADAPTER_TARGET = "llm-adapter-target"
NON_DIFFUSION_OWNER_BUNDLE = "non-diffusion-owner-bundle"
REGIONAL_HOOK_FIXTURES = (
    NON_WEIGHT_HOOK,
    MODEL_AS_LORA_FORMAT,
    UNSUPPORTED_SUFFIX_FORMAT,
    INCOMPLETE_STANDARD_PAIR,
    UNSUPPORTED_ANIMA_TARGET,
    LLM_ADAPTER_TARGET,
    NON_DIFFUSION_OWNER_BUNDLE,
)

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class CreateRegionalHookFixtureV3(_ComfyNodeBase):
    """Expose one closed rejection-only HookGroup fixture."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the dev-only fixture selector and HOOKS output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.CreateRegionalHookFixture",
            display_name="Benchmark Create Regional Hook Fixture",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Combo.Input(
                    "fixture",
                    options=list(REGIONAL_HOOK_FIXTURES),
                    default=NON_WEIGHT_HOOK,
                )
            ],
            outputs=[_comfy_io.Hooks.Output("hooks")],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, fixture: str) -> Any:
        """Return one newly constructed exact installed-Comfy HookGroup."""

        return _comfy_io.NodeOutput(create_regional_hook_fixture(fixture))


def create_regional_hook_fixture(fixture: str) -> comfy.hooks.HookGroup:
    """Construct one deterministic CPU-only rejection fixture."""

    if fixture == NON_WEIGHT_HOOK:
        hooks = comfy.hooks.HookGroup()
        hooks.add(comfy.hooks.TransformerOptionsHook({}))
        return hooks
    if fixture == MODEL_AS_LORA_FORMAT:
        return comfy.hooks.create_hook_model_as_lora(
            weights_model={
                "diffusion_model.blocks.0.self_attn.q_proj.weight": torch.ones(
                    (1,), dtype=torch.float32
                )
            },
            weights_clip=None,
            strength_model=1.0,
            strength_clip=0.0,
        )
    if fixture == UNSUPPORTED_SUFFIX_FORMAT:
        return _raw_weight_hook(
            {
                "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight": (
                    torch.ones((1, 2048), dtype=torch.float32)
                ),
                "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight": (
                    torch.ones((2048, 1), dtype=torch.float32)
                ),
            }
        )
    if fixture == INCOMPLETE_STANDARD_PAIR:
        return _raw_weight_hook(
            {
                "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": (
                    torch.ones((1, 2048), dtype=torch.float32)
                )
            }
        )
    if fixture == UNSUPPORTED_ANIMA_TARGET:
        return _raw_weight_hook(
            {
                "diffusion_model.blocks.0.unknown_projection.lora_A.weight": (
                    torch.ones((1, 2048), dtype=torch.float32)
                ),
                "diffusion_model.blocks.0.unknown_projection.lora_B.weight": (
                    torch.ones((2048, 1), dtype=torch.float32)
                ),
            }
        )
    if fixture == LLM_ADAPTER_TARGET:
        target = "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj"
        return _raw_weight_hook(
            {
                f"{target}.lora_A.weight": torch.ones((1, 1024), dtype=torch.float32),
                f"{target}.lora_B.weight": torch.ones((1024, 1), dtype=torch.float32),
            }
        )
    if fixture == NON_DIFFUSION_OWNER_BUNDLE:
        return _raw_weight_hook(
            {
                "lora_te_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                    (1, 1024), dtype=torch.float32
                ),
                "lora_te_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                    (1024, 1), dtype=torch.float32
                ),
                "vae.decoder.conv_in.lora_A.weight": torch.ones(
                    (1, 16), dtype=torch.float32
                ),
                "vae.decoder.conv_in.lora_B.weight": torch.ones(
                    (128, 1), dtype=torch.float32
                ),
            }
        )
    raise ValueError(f"Unknown regional hook fixture: {fixture!r}.")


def _raw_weight_hook(weights: dict[str, torch.Tensor]) -> comfy.hooks.HookGroup:
    """Retain one exact raw loader mapping in an installed WeightHook."""

    return comfy.hooks.create_hook_lora(
        lora=weights,
        strength_model=1.0,
        strength_clip=0.0,
    )
