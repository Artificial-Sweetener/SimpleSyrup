# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration tests for native Comfy hook-aware regional evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import comfy.conds
import comfy.hooks
import comfy.sampler_helpers
import comfy.samplers
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.domain.tiled_diffusion import build_tiled_diffusion_plan
from simple_syrup.runtime.contextual_model_wrapper import (
    ContextualDiffusionModelWrapper,
)
from simple_syrup.runtime.mixture_of_diffusers_sampling import (
    MixtureOfDiffusersModelWrapper,
)
from simple_syrup.runtime.multidiffusion_sampling import MultiDiffusionModelWrapper
from simple_syrup.services.regional_conditioning_service import (
    RegionalConditioningService,
)

ModelWrapperFactory = Callable[[], object]


class FakePatcher:
    """Record hook preparation and activation performed by Comfy's sampler."""

    def __init__(self) -> None:
        """Initialize hook observations for one model evaluation."""

        self.prepared: list[comfy.hooks.HookGroup] = []
        self.applied: list[comfy.hooks.HookGroup | None] = []
        self.active: comfy.hooks.HookGroup | None = None

    def prepare_hook_patches_current_keyframe(
        self,
        timestep: torch.Tensor,
        hooks: comfy.hooks.HookGroup,
        model_options: dict[str, Any],
    ) -> None:
        """Record the hook group prepared for the current sigma."""

        del timestep, model_options
        self.prepared.append(hooks)

    def prepare_state(
        self,
        timestep: torch.Tensor,
        model_options: dict[str, Any],
    ) -> None:
        """Accept Comfy's per-step patcher state preparation."""

        del timestep, model_options

    def get_free_memory(self, device: torch.device) -> float:
        """Return ample deterministic capacity for conditioning batches."""

        del device
        return 1.0e12

    def apply_hooks(
        self,
        *,
        hooks: comfy.hooks.HookGroup | None,
    ) -> dict[str, Any]:
        """Record the hook group activated before the model call."""

        self.applied.append(hooks)
        self.active = hooks
        return {}


class FakeModel:
    """Provide the minimal model surface used by native conditioning evaluation."""

    def __init__(self) -> None:
        """Create a model with an observable hook-aware patcher."""

        self.current_patcher = FakePatcher()
        self.model_calls: list[comfy.hooks.HookGroup | None] = []

    def memory_required(
        self,
        input_shape: list[int],
        *,
        cond_shapes: dict[str, list[list[int]]],
    ) -> float:
        """Return a fixed estimate so Comfy can select a batch size."""

        del input_shape, cond_shapes
        return 1.0

    def apply_model(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: Any,
    ) -> torch.Tensor:
        """Record the active hook group and return a deterministic prediction."""

        del timestep, conditioning
        self.model_calls.append(self.current_patcher.active)
        return torch.ones_like(input_x)


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        None,
        lambda: MultiDiffusionModelWrapper(
            plan=build_tiled_diffusion_plan(
                latent_width=32,
                latent_height=32,
                tile_width=16,
                tile_height=16,
                overlap=2,
                tile_batch_size=2,
            ),
            existing_wrapper=None,
        ),
        lambda: MixtureOfDiffusersModelWrapper(
            plan=build_tiled_diffusion_plan(
                latent_width=32,
                latent_height=32,
                tile_width=16,
                tile_height=16,
                overlap=2,
                tile_batch_size=2,
            ),
            existing_wrapper=None,
        ),
        lambda: _contextual_wrapper(),
    ],
    ids=["native", "multidiffusion", "mixture-of-diffusers", "contextual"],
)
def test_native_sampler_activates_hooks_from_masked_regional_conditioning(
    wrapper_factory: ModelWrapperFactory | None,
) -> None:
    """Comfy activates each preserved hook group through direct and tiled paths."""

    global_hooks = comfy.hooks.HookGroup()
    regional_hooks = comfy.hooks.HookGroup()
    assembled, _ = RegionalConditioningService().assemble(
        positive=ConditioningBatch(
            (
                _conditioning(global_hooks),
                _conditioning(regional_hooks),
            )
        ),
        negative=_conditioning(None),
        masks=torch.ones((1, 32, 32)),
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )
    converted = comfy.sampler_helpers.convert_cond(assembled)
    for conditioning in converted:
        cross_attn = conditioning.pop("cross_attn")
        conditioning["model_conds"] = {
            "c_crossattn": comfy.conds.CONDCrossAttn(cross_attn)
        }

    model = FakeModel()
    model_options: dict[str, Any] = {}
    if wrapper_factory is not None:
        model_options["model_function_wrapper"] = wrapper_factory()

    outputs = comfy.samplers.calc_cond_batch(
        model,
        [converted],
        torch.zeros((1, 1, 32, 32)),
        torch.ones((1,)),
        model_options,
    )

    assert len(outputs) == 1
    assert set(model.current_patcher.prepared) == {global_hooks, regional_hooks}
    assert set(model.current_patcher.applied) == {global_hooks, regional_hooks}
    assert set(model.model_calls) == {global_hooks, regional_hooks}


def _conditioning(
    hooks: comfy.hooks.HookGroup | None,
) -> list[list[object]]:
    """Return one raw Comfy conditioning entry with optional hooks."""

    metadata: dict[str, object] = {}
    if hooks is not None:
        metadata["hooks"] = hooks
    return [[torch.ones((1, 2, 3)), metadata]]


def _contextual_wrapper() -> ContextualDiffusionModelWrapper:
    """Return a multi-view Contextual wrapper with active global authority."""

    controls = ContextualDiffusionControls(16, 2, 2, 1.0, 1, 0.5)
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=32,
        controls=controls,
        segs=None,
    )
    return ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )
