"""Verify regional-detailing masks and bounds at reduced inversion resolutions."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_detailing import LatentBox, LatentRegion
from simple_syrup.domain.regional_inversion_geometry import project_inversion_regions
from simple_syrup.runtime.regional_inversion_model_factory import (
    RegionalInversionModelFactory,
)
from simple_syrup.runtime.regional_multidiffusion_prediction import (
    RegionalMultiDiffusionCalcCondBatch,
)

from .support.inversion_model import SpatialModel


def region() -> LatentRegion:
    """Keep an asymmetric region with canonical full-canvas ownership."""
    mask = torch.zeros((9, 13))
    mask[2:8, 3:12] = 1
    return LatentRegion(7, "portrait", LatentBox(3, 2, 9, 6), mask, [])


def test_full_resolution_preserves_original_region_identity() -> None:
    """Reuse canonical geometry without resampling during full-size finishing."""
    regions = (region(),)
    assert (
        project_inversion_regions(
            regions,
            source_width=13,
            source_height=9,
            target_width=13,
            target_height=9,
        )
        is regions
    )


def test_coarse_projection_preserves_conditioning_and_coverage() -> None:
    """Cover the scaled bounds while retaining the exact regional prompt payload."""
    original = region()
    (projected,) = project_inversion_regions(
        (original,),
        source_width=13,
        source_height=9,
        target_width=6,
        target_height=4,
    )
    assert projected.index == original.index and projected.label == original.label
    assert projected.positive is original.positive
    assert projected.latent_box == LatentBox(1, 0, 5, 4)
    assert projected.latent_mask.shape == (4, 6)
    assert bool(torch.all((projected.latent_mask == 0) | (projected.latent_mask == 1)))
    assert bool(torch.any(projected.latent_mask == 1))
    assert original.latent_mask.shape == (9, 13)


@pytest.mark.parametrize("bad_box", [LatentBox(-1, 0, 2, 2), LatentBox(0, 0, 14, 2)])
def test_invalid_canonical_bounds_fail_closed(bad_box: LatentBox) -> None:
    """Do not silently clamp malformed ownership into another region."""
    original = region()
    malformed = LatentRegion(7, "portrait", bad_box, original.latent_mask, [])
    with pytest.raises(ValueError, match="bounds"):
        project_inversion_regions(
            (malformed,),
            source_width=13,
            source_height=9,
            target_width=6,
            target_height=4,
        )


def test_mismatched_mask_canvas_fails_closed() -> None:
    """Reject masks from a different coordinate system before stage execution."""
    with pytest.raises(ValueError, match="canonical canvas"):
        project_inversion_regions(
            (region(),),
            source_width=12,
            source_height=9,
            target_width=6,
            target_height=4,
        )


def test_factory_rebuilds_each_stage_from_the_unwrapped_model() -> None:
    """Install real regional wrappers without nesting a forward regional wrapper."""
    original = SpatialModel()
    factory = RegionalInversionModelFactory(
        model=original,
        canvas_width=13,
        canvas_height=9,
        regions=(region(),),
        global_prompt_weight=0.3,
        differential_diffusion=False,
    )
    for height, width in ((4, 6), (9, 13)):
        derived = factory(torch.zeros((1, 4, height, width)))
        assert derived.parent is original
        wrapper: Any = derived.model_options["sampler_calc_cond_batch_function"]
        assert isinstance(wrapper, RegionalMultiDiffusionCalcCondBatch)
        assert wrapper._regions[0].latent_mask.shape == (height, width)
    assert original.model_options == {}
