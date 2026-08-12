"""Characterize Regional Conditioning across sampling layouts and tile batches."""

from collections.abc import Callable
from typing import Any, cast

import comfy.conds
import comfy.sampler_helpers
import comfy.samplers
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
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

ModelFunction = Callable[..., torch.Tensor]
ModelFunctionWrapper = Callable[[ModelFunction, dict[str, Any]], torch.Tensor]


class _FakePatcher:
    """Provide the hook lifecycle required by Comfy's conditioning evaluator."""

    def prepare_hook_patches_current_keyframe(
        self,
        timestep: torch.Tensor,
        hooks: object,
        model_options: dict[str, Any],
    ) -> None:
        """Accept Comfy's current-keyframe preparation call."""
        del timestep, hooks, model_options

    def prepare_state(
        self,
        timestep: torch.Tensor,
        model_options: dict[str, Any],
    ) -> None:
        """Accept Comfy's hook-state preparation call."""
        del timestep, model_options

    def get_free_memory(self, device: torch.device) -> float:
        """Report ample memory so compatible conditions may batch together."""
        del device
        return 1e12

    def apply_hooks(
        self,
        *,
        hooks: object | None,
    ) -> dict[str, torch.Tensor]:
        """Accept hook activation without introducing model patches."""
        del hooks
        return {}


class _RegionalPredictionModel:
    """Evaluate regional conditioning while recording runtime batch shapes."""

    def __init__(self, wrapper: ModelFunctionWrapper | None = None) -> None:
        """Initialize the deterministic prediction model."""
        self.current_patcher = _FakePatcher()
        self.wrapper = wrapper
        self.calls: list[tuple[int, ...]] = []

    def memory_required(
        self,
        input_shape: tuple[int, ...] | list[int],
        **requirements: Any,
    ) -> float:
        """Report a small deterministic memory requirement."""
        del input_shape, requirements
        return 1.0

    def apply_model(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: Any,
    ) -> torch.Tensor:
        """Apply the selected sampling wrapper around the deterministic model."""
        if self.wrapper is None:
            return self._evaluate(input_x, timestep, **conditioning)
        return self.wrapper(
            self._evaluate,
            {"input": input_x, "timestep": timestep, "c": conditioning},
        )

    def _evaluate(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: Any,
    ) -> torch.Tensor:
        """Turn each cross-attention value into a spatially constant prediction."""
        del timestep
        self.calls.append(tuple(input_x.shape))
        cross_attention = conditioning["c_crossattn"]
        if not isinstance(cross_attention, torch.Tensor):
            raise TypeError("Cross-attention conditioning must be a tensor.")
        reduction_dimensions = tuple(range(1, cross_attention.ndim))
        values = cross_attention.float().mean(dim=reduction_dimensions)
        value_shape = (-1,) + (1,) * (input_x.ndim - 1)
        return torch.ones_like(input_x) * values.reshape(value_shape)


def _conditioning(value: float) -> list[list[object]]:
    """Build one raw conditioning entry with a recognizable prediction value."""
    cross_attention = torch.full((1, 2, 3), value, dtype=torch.float32)
    return [[cross_attention, {}]]


def _regional_conditioning(mask: torch.Tensor) -> list[dict[str, Any]]:
    """Build converted global-plus-regional conditioning for Comfy evaluation."""
    service = RegionalConditioningService()
    positive = ConditioningBatch((_conditioning(1.0), _conditioning(3.0)))
    negative = ConditioningBatch((_conditioning(-1.0), _conditioning(-3.0)))
    regional_positive, _regional_negative = service.assemble(
        positive=positive,
        negative=negative,
        masks=mask,
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )
    converted = cast(
        list[dict[str, Any]], comfy.sampler_helpers.convert_cond(regional_positive)
    )
    for entry in converted:
        cross_attention = entry.pop("cross_attn")
        entry["model_conds"] = {
            "c_crossattn": comfy.conds.CONDCrossAttn(cross_attention)
        }
    return converted


def _samples(layout_dimensions: int) -> torch.Tensor:
    """Create a BCHW or singleton-depth BCDHW latent."""
    if layout_dimensions == 4:
        return torch.zeros((1, 1, 32, 64), dtype=torch.float32)
    return torch.zeros((1, 1, 1, 32, 64), dtype=torch.float32)


def _left_half_mask() -> torch.Tensor:
    """Create a mask selecting the left half of the latent plane."""
    mask = torch.zeros((1, 32, 64), dtype=torch.float32)
    mask[:, :, :32] = 1.0
    return mask


def _expected_prediction(samples: torch.Tensor) -> torch.Tensor:
    """Create the blended prediction expected from the regional mask."""
    expected = torch.ones_like(samples)
    expected[..., :32] = 2.0
    return expected


def _predict(
    samples: torch.Tensor,
    wrapper: ModelFunctionWrapper | None,
) -> tuple[torch.Tensor, _RegionalPredictionModel]:
    """Evaluate Regional Conditioning through Comfy's native condition path."""
    return _predict_conditioning(
        samples,
        _regional_conditioning(_left_half_mask()),
        wrapper,
    )


def _predict_conditioning(
    samples: torch.Tensor,
    conditioning: list[dict[str, Any]],
    wrapper: ModelFunctionWrapper | None,
) -> tuple[torch.Tensor, _RegionalPredictionModel]:
    """Evaluate one converted regional conditioning through an optional wrapper."""

    model = _RegionalPredictionModel(wrapper)
    prediction = comfy.samplers.calc_cond_batch(
        model,
        [conditioning],
        samples,
        torch.ones((samples.shape[0],), dtype=torch.float32),
        {},
    )[0]
    return prediction, model


@pytest.mark.parametrize("layout_dimensions", [4, 5])
def test_full_sampling_preserves_regional_prediction(
    layout_dimensions: int,
) -> None:
    """Preserve regional blending for full BCHW and BCDHW sampling."""
    samples = _samples(layout_dimensions)

    prediction, model = _predict(samples, None)

    torch.testing.assert_close(prediction, _expected_prediction(samples))
    assert prediction.shape == samples.shape
    assert model.calls == [(2, *samples.shape[1:])]


@pytest.mark.parametrize(
    "wrapper_type",
    [MultiDiffusionModelWrapper, MixtureOfDiffusersModelWrapper],
)
@pytest.mark.parametrize("layout_dimensions", [4, 5])
@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
def test_tiled_sampling_preserves_regional_prediction(
    wrapper_type: type[MultiDiffusionModelWrapper]
    | type[MixtureOfDiffusersModelWrapper],
    layout_dimensions: int,
    tile_batch_size: int,
) -> None:
    """Preserve regional blending across tiled modes, layouts, and batch sizes."""
    plan = build_tiled_diffusion_plan(
        latent_width=64,
        latent_height=32,
        tile_width=16,
        tile_height=16,
        overlap=0,
        tile_batch_size=tile_batch_size,
    )
    wrapper = wrapper_type(plan=plan, existing_wrapper=None)
    samples = _samples(layout_dimensions)

    prediction, model = _predict(samples, wrapper)

    torch.testing.assert_close(prediction, _expected_prediction(samples))
    assert prediction.shape == samples.shape
    assert max(call[0] for call in model.calls) == tile_batch_size * 2


@pytest.mark.parametrize("diffusion_mode", ["multidiffusion", "mixture_of_diffusers"])
@pytest.mark.parametrize("layout_dimensions", [4, 5])
@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
def test_contextual_sampling_preserves_regional_prediction(
    diffusion_mode: str,
    layout_dimensions: int,
    tile_batch_size: int,
) -> None:
    """Preserve regional blending across contextual modes and layouts."""
    controls = ContextualDiffusionControls(
        latent_context_size=16,
        latent_context_overlap=0,
        latent_context_batch_size=tile_batch_size,
        global_weight=1.0,
        global_steps=1,
        global_decay=0.5,
    )
    plan = build_contextual_diffusion_plan(
        latent_width=64,
        latent_height=32,
        controls=controls,
        segs=None,
        region_masks=None,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0], dtype=torch.float32),
        diffusion_mode=diffusion_mode,
        existing_wrapper=None,
    )
    samples = _samples(layout_dimensions)

    prediction, model = _predict(samples, wrapper)

    torch.testing.assert_close(prediction, _expected_prediction(samples))
    assert prediction.shape == samples.shape
    assert max(call[0] for call in model.calls) == tile_batch_size * 2


@pytest.mark.parametrize("diffusion_mode", ["multidiffusion", "mixture_of_diffusers"])
@pytest.mark.parametrize("layout_dimensions", [4, 5])
@pytest.mark.parametrize("use_segs", [False, True], ids=["grid", "segs"])
def test_contextual_sampling_preserves_overlapping_and_uncovered_regions(
    diffusion_mode: str,
    layout_dimensions: int,
    use_segs: bool,
) -> None:
    """Match native regional overlap and fallback across grid and SEGS plans."""

    masks = _overlap_and_uncovered_masks()
    conditioning = _regional_conditioning_for_masks(masks)
    samples = _samples(layout_dimensions)
    expected, _direct_model = _predict_conditioning(samples, conditioning, None)
    controls = ContextualDiffusionControls(16, 4, 2, 1.0, 1, 0.5)
    plan = build_contextual_diffusion_plan(
        latent_width=64,
        latent_height=32,
        controls=controls,
        segs=_context_segs() if use_segs else None,
        region_masks=masks,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0], dtype=torch.float32),
        diffusion_mode=diffusion_mode,
        existing_wrapper=None,
    )

    prediction, model = _predict_conditioning(samples, conditioning, wrapper)

    torch.testing.assert_close(prediction, expected)
    assert prediction.shape == samples.shape
    assert all(tile.weight_mask is not None for tile in plan.tile_plan.tiles)
    assert len(model.calls) > len(_direct_model.calls)


def _overlap_and_uncovered_masks() -> torch.Tensor:
    """Create two overlapping regions with uncovered strips on both sides."""

    masks = torch.zeros((2, 32, 64), dtype=torch.float32)
    masks[0, :, 8:36] = 1.0
    masks[1, :, 28:56] = 1.0
    return masks


def _regional_conditioning_for_masks(
    masks: torch.Tensor,
) -> list[dict[str, Any]]:
    """Build converted global, first-region, and second-region conditioning."""

    positive, _negative = RegionalConditioningService().assemble(
        positive=ConditioningBatch(
            (_conditioning(1.0), _conditioning(3.0), _conditioning(5.0))
        ),
        negative=_conditioning(-1.0),
        masks=masks,
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )
    converted = cast(list[dict[str, Any]], comfy.sampler_helpers.convert_cond(positive))
    for entry in converted:
        cross_attention = entry.pop("cross_attn")
        entry["model_conds"] = {
            "c_crossattn": comfy.conds.CONDCrossAttn(cross_attention)
        }
    return converted


def _context_segs() -> tuple[tuple[int, int], tuple[Segment, ...]]:
    """Create one central SEGS constraint over the regional planning canvas."""

    mask = torch.zeros((32, 64), dtype=torch.float32)
    mask[4:28, 12:52] = 1.0
    crop = CropRegion(0, 0, 64, 32)
    segment = Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=1.0,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="context",
    )
    return (32, 64), (segment,)
