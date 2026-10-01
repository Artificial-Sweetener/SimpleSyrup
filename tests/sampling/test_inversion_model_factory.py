"""Prove resolution-specific inversion planning and canonical regional masks."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import ContextualDiffusionControls
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.sampler_options import TilingOptions
from simple_syrup.domain.spatial_views import SpatialBatchLayout, SpatialViewKind
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
    RegionalMaskProjector,
)
from simple_syrup.runtime.contextual_model_wrapper import (
    ContextualDiffusionModelWrapper,
)
from simple_syrup.runtime.inversion_model_factory import InversionModelFactory
from simple_syrup.runtime.inversion_spatial_context import (
    InversionSpatialContextWrapper,
)

from .support.inversion_model import SpatialModel


@pytest.mark.parametrize("mode", ("multidiffusion", "mixture_of_diffusers"))
@pytest.mark.parametrize("contextual", (False, True))
def test_each_stage_is_planned_from_the_original_model(
    mode: str, contextual: bool
) -> None:
    """Avoid full-size fallback, nested tiling and mutation of the supplied MODEL."""
    model = SpatialModel()
    context = ContextualDiffusionControls(16, 4, 2, 0.5, 1, 0.5)
    factory = InversionModelFactory(
        model=model,
        canvas_width=64,
        canvas_height=32,
        tiling=TilingOptions(
            diffusion_mode=mode, width=16, height=16, overlap=4, batch_size=2
        ),
        context=context if contextual else None,
        forward_sigmas=torch.tensor([0.5, 0.25, 0]),
    )
    model_call_shapes: list[tuple[int, ...]] = []

    def denoiser(x: torch.Tensor, timestep: torch.Tensor, **c: Any) -> torch.Tensor:
        """Provide only the external neural boundary with a deterministic prediction."""
        model_call_shapes.append(tuple(x.shape))
        return x + 1

    for height, width in ((16, 32), (32, 64)):
        stage_model = factory(torch.zeros((1, 4, height, width)))
        assert stage_model.parent is model
        wrapper = stage_model.model_options["model_function_wrapper"]
        if contextual:
            assert isinstance(wrapper, ContextualDiffusionModelWrapper)
        start = len(model_call_shapes)
        x = torch.zeros((1, 4, height, width))
        result = wrapper(
            denoiser, {"input": x, "timestep": torch.tensor([0.5]), "c": {}}
        )
        assert result.shape == x.shape
        assert model_call_shapes[start:]
        assert all(
            shape[-2] <= 16 and shape[-1] <= 16 for shape in model_call_shapes[start:]
        )
    assert model.model_options == {}


@pytest.mark.parametrize("contextual", (False, True))
def test_regional_mask_bank_keeps_original_coordinates_in_coarse_views(
    contextual: bool,
) -> None:
    """Use real mask projection after coarse local or global context calls."""
    mask = torch.zeros((1, 32, 64))
    mask[:, :, 32:] = 1
    bank = RegionalMaskBank(mask, mask.clone(), 64, 32)
    layouts: list[SpatialBatchLayout] = []
    options: list[dict[str, Any]] = []

    def original(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
        """Exercise the real regional projection authority at the preserved wrapper."""
        transformer = args["c"]["transformer_options"]
        layout = transformer["simple_syrup"]["spatial_batch_layout"]
        assert isinstance(layout, SpatialBatchLayout)
        assert (layout.canvas_width, layout.canvas_height) == (64, 32)
        assert layout.expanded_batch_size == int(args["input"].shape[0])
        for index in range(layout.view_count):
            projected = RegionalMaskProjector().project_view(
                bank=bank,
                layout=layout,
                view_index=index,
                form=RegionalMaskForm.CONDITIONING,
                mode=RegionalMaskProjectionMode.NEAREST,
            )
            assert tuple(projected.shape[-2:]) == (
                layout.views[index].model_height,
                layout.views[index].model_width,
            )
        layouts.append(layout)
        options.append(transformer)
        prediction_input = args["input"]
        assert isinstance(prediction_input, torch.Tensor)
        return prediction_input + 1

    model = SpatialModel({"model_function_wrapper": original})
    factory = InversionModelFactory(
        model=model,
        canvas_width=64,
        canvas_height=32,
        tiling=TilingOptions(width=16, height=16, overlap=4, batch_size=2),
        context=ContextualDiffusionControls(16, 4, 2, 1.0, 1, 0.5)
        if contextual
        else None,
        forward_sigmas=torch.tensor([0.5, 0.25, 0]),
        region_masks=mask if contextual else None,
    )
    stage = factory(torch.zeros((2, 4, 16, 32)))
    wrapper = stage.model_options["model_function_wrapper"]
    transformer = {"cond_or_uncond": [0, 1], "uuids": ["positive", "negative"]}
    args = {
        "input": torch.zeros((2, 4, 16, 32)),
        "timestep": torch.tensor([0.5, 0.5]),
        "c": {"transformer_options": transformer},
    }
    wrapper(lambda *args, **kwargs: None, args)
    assert layouts
    assert any(layout.views[0].kind is SpatialViewKind.TILE for layout in layouts)
    if contextual:
        assert any(
            layout.views[0].kind is SpatialViewKind.CONTEXTUAL_GLOBAL
            for layout in layouts
        )
    for layout, transformed in zip(layouts, options, strict=True):
        assert len(transformed["cond_or_uncond"]) == 2 * layout.view_count
    assert transformer == {"cond_or_uncond": [0, 1], "uuids": ["positive", "negative"]}
    assert model.model_options["model_function_wrapper"] is original


def test_full_frame_and_single_tile_calls_publish_whole_source_reduced_layout() -> None:
    """Preserve regional attention through the one-tile fast path."""
    layouts: list[SpatialBatchLayout] = []

    def original(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
        """Inspect only the host wrapper boundary's spatial metadata."""
        layout = args["c"]["transformer_options"]["simple_syrup"][
            "spatial_batch_layout"
        ]
        layouts.append(layout)
        prediction_input = args["input"]
        assert isinstance(prediction_input, torch.Tensor)
        return prediction_input

    model = SpatialModel({"model_function_wrapper": original})
    for tiling in (None, TilingOptions(width=32, height=32, overlap=4)):
        factory = InversionModelFactory(
            model=model, canvas_width=48, canvas_height=32, tiling=tiling
        )
        stage = factory(torch.zeros((1, 4, 16, 24)))
        stage.model_options["model_function_wrapper"](
            lambda *args, **kwargs: None,
            {
                "input": torch.zeros((1, 4, 16, 24)),
                "timestep": torch.tensor([0.5]),
                "c": {},
            },
        )
    assert len(layouts) == 2
    for layout in layouts:
        assert (layout.canvas_width, layout.canvas_height) == (48, 32)
        assert layout.views[0].kind is SpatialViewKind.CONTEXTUAL_GLOBAL
        assert (layout.views[0].model_width, layout.views[0].model_height) == (24, 16)


def test_malformed_reduced_layout_is_not_guessed() -> None:
    """Reject calls missing the metadata needed to identify their source crop."""
    wrapper = InversionSpatialContextWrapper(
        lambda apply_model, args: args["input"],
        canvas_width=64,
        canvas_height=32,
        stage_width=32,
        stage_height=16,
    )
    with pytest.raises(ValueError, match="explicit spatial layout"):
        wrapper(
            lambda *args, **kwargs: torch.empty(0),
            {"input": torch.zeros((1, 4, 8, 8)), "c": {}},
        )
