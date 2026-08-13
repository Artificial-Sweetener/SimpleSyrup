# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare regional attention weighting with installed Comfy default conditions."""

from __future__ import annotations

from typing import Any, cast

import comfy.conds
import comfy.sampler_helpers
import comfy.samplers
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
)
from simple_syrup.masking.regional_prompt_masks import build_regional_mask_bank
from simple_syrup.services.regional_conditioning_service import (
    RegionalConditioningService,
)


class _ComfyHarnessPatcher:
    """Provide the hook lifecycle required by Comfy condition evaluation."""

    def prepare_hook_patches_current_keyframe(
        self,
        timestep: torch.Tensor,
        hooks: object,
        model_options: dict[str, Any],
    ) -> None:
        """Accept the installed Comfy hook-keyframe boundary."""

        del timestep, hooks, model_options

    def prepare_state(
        self,
        timestep: torch.Tensor,
        model_options: dict[str, Any],
    ) -> None:
        """Accept the installed Comfy hook-state boundary."""

        del timestep, model_options

    def get_free_memory(self, device: torch.device) -> float:
        """Report enough memory for Comfy to batch compatible conditions."""

        del device
        return 1e12

    def apply_hooks(self, *, hooks: object | None) -> dict[str, torch.Tensor]:
        """Return no model patches for the deterministic harness."""

        del hooks
        return {}


class _ConstantConditionModel:
    """Map each cross-attention branch to its recognizable scalar value."""

    def __init__(self) -> None:
        """Create the lifecycle adapter expected by Comfy's sampler."""

        self.current_patcher = _ComfyHarnessPatcher()

    def memory_required(
        self,
        input_shape: tuple[int, ...] | list[int],
        **requirements: Any,
    ) -> float:
        """Report one deterministic small model-memory estimate."""

        del input_shape, requirements
        return 1.0

    def apply_model(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: Any,
    ) -> torch.Tensor:
        """Return one spatially constant output per conditioning branch."""

        del timestep
        cross_attention = conditioning["c_crossattn"]
        if not isinstance(cross_attention, torch.Tensor):
            raise TypeError("Comfy harness cross-attention must be a torch.Tensor.")
        reduction_dimensions = tuple(range(1, cross_attention.ndim))
        values = cross_attention.float().mean(dim=reduction_dimensions)
        output_shape = (-1,) + (1,) * (input_x.ndim - 1)
        return torch.ones_like(input_x) * values.reshape(output_shape)


def _partial_mask() -> torch.Tensor:
    """Return one region with hard partial coverage."""

    masks = torch.zeros((1, 4, 6))
    masks[:, :, :3] = 1.0
    return masks


def _overlapping_masks() -> torch.Tensor:
    """Return two regions whose combined weighted coverage exceeds one."""

    masks = torch.zeros((2, 4, 6))
    masks[0, :, :4] = 1.0
    masks[1, :, 2:] = 1.0
    return masks


def _uncovered_masks() -> torch.Tensor:
    """Return two regions that leave the center query columns to the base."""

    masks = torch.zeros((2, 4, 6))
    masks[0, :, :2] = 1.0
    masks[1, :, 4:] = 1.0
    return masks


def _feathered_mask() -> torch.Tensor:
    """Return one soft conditioning form built by the canonical mask factory."""

    authored = torch.zeros((1, 4, 6))
    authored[:, 1:3, 2:4] = 1.0
    return build_regional_mask_bank(
        authored,
        feather=1,
        canvas_height=4,
        canvas_width=6,
    ).conditioning_masks


@pytest.mark.parametrize(
    ("case_name", "masks"),
    [
        ("zero", torch.zeros((1, 4, 6))),
        ("solid", torch.ones((1, 4, 6))),
        ("partial", _partial_mask()),
        ("overlapping", _overlapping_masks()),
        ("uncovered", _uncovered_masks()),
        ("feathered", _feathered_mask()),
    ],
)
def test_weighting_matches_installed_comfy_default_complement(
    case_name: str,
    masks: torch.Tensor,
) -> None:
    """Match real Comfy accumulation for every required regional mask shape."""

    del case_name
    regional_strength = 0.75
    base_value = 1.0
    regional_values = tuple(3.0 + 2.0 * index for index in range(int(masks.shape[0])))
    comfy_output = _comfy_prediction(
        masks=masks,
        base_value=base_value,
        regional_values=regional_values,
        regional_strength=regional_strength,
    )

    policy = RegionalAttentionWeightingPolicy()
    weights = policy.weights(
        masks,
        region_strengths=(regional_strength,) * int(masks.shape[0]),
    )
    base_output = torch.full(masks.shape[1:], base_value)
    regional_outputs = torch.stack(
        [torch.full(masks.shape[1:], value) for value in regional_values]
    )
    policy_output = policy.blend(
        weights=weights,
        base_output=base_output,
        regional_outputs=regional_outputs,
    )

    torch.testing.assert_close(policy_output, comfy_output[0, 0])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_strength_fails_before_mask_device_value_inspection(
    invalid: float,
) -> None:
    """Reject invalid scalar admission before inspecting an invalid mask tensor."""

    masks = torch.tensor([[float("nan")]])

    with pytest.raises(ValueError, match="strength 0 must be finite"):
        RegionalAttentionWeightingPolicy().weights(
            masks,
            region_strengths=(invalid,),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_mask_fails_before_strength_tensor_construction(invalid: float) -> None:
    """Reject invalid coverage before materializing strengths on its device."""

    with pytest.raises(ValueError, match="masks must contain finite values"):
        RegionalAttentionWeightingPolicy().weights(
            torch.tensor([[invalid]]),
            region_strengths=(1.0,),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_branch_output_fails_before_blend_arithmetic(invalid: float) -> None:
    """Reject invalid branch results before weighted multiplication or division."""

    policy = RegionalAttentionWeightingPolicy()
    weights = policy.weights(torch.ones((1, 1)), region_strengths=(1.0,))

    with pytest.raises(ValueError, match="outputs must contain finite values"):
        policy.blend(
            weights=weights,
            base_output=torch.tensor([0.0]),
            regional_outputs=torch.tensor([[invalid]]),
        )


def _comfy_prediction(
    *,
    masks: torch.Tensor,
    base_value: float,
    regional_values: tuple[float, ...],
    regional_strength: float,
) -> torch.Tensor:
    """Evaluate ordinary masked/default conditions through installed Comfy."""

    entries = (_conditioning(base_value),) + tuple(
        _conditioning(value) for value in regional_values
    )
    assembled, _ = RegionalConditioningService().assemble_prepared(
        positive=ConditioningBatch(entries),
        negative=_conditioning(-1.0),
        mask_batch=masks,
        regional_prompt_weight=regional_strength,
    )
    converted = cast(
        list[dict[str, Any]], comfy.sampler_helpers.convert_cond(assembled)
    )
    for entry in converted:
        cross_attention = entry.pop("cross_attn")
        entry["model_conds"] = {
            "c_crossattn": comfy.conds.CONDCrossAttn(cross_attention)
        }
    samples = torch.zeros((1, 1, *masks.shape[1:]))
    prediction = comfy.samplers.calc_cond_batch(
        _ConstantConditionModel(),
        [converted],
        samples,
        torch.ones((1,)),
        {},
    )[0]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Comfy harness prediction must be a torch.Tensor.")
    return prediction


def _conditioning(value: float) -> list[list[object]]:
    """Return one raw conditioning branch with a recognizable scalar value."""

    return [[torch.full((1, 2, 3), value), {}]]
