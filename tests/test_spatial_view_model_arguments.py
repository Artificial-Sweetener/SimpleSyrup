"""Verify apply-model argument transformation across spatial views."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.spatial_model_arguments import make_spatial_view_model_args


def _left_right_views() -> tuple[SpatialView, SpatialView]:
    """Return two equal-model-shape views spanning an eight-column canvas."""

    return (
        SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
        SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 4, 4),
    )


def test_spatial_args_reject_unequal_view_model_shapes() -> None:
    """Require one shared model shape for a batched model call."""

    args = {"input": torch.zeros((1, 1, 4, 8)), "timestep": torch.ones((1,)), "c": {}}
    unequal = (
        SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
        SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 2, 4),
    )
    with pytest.raises(ValueError, match="one model spatial shape"):
        make_spatial_view_model_args(
            args=args,
            layout=_layout(unequal, input_batch_size=1),
        )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            {"input": "tensor", "timestep": torch.ones((1,)), "c": {}},
            "model input must be a tensor",
        ),
        (
            {"input": torch.zeros((1, 1, 4, 8)), "timestep": 1.0, "c": {}},
            "timestep must be a tensor",
        ),
        (
            {
                "input": torch.zeros((1, 1, 4, 8)),
                "timestep": torch.ones((1,)),
                "c": [],
            },
            "conditioning must be a dict",
        ),
    ],
)
def test_spatial_args_reject_dynamic_model_boundaries(
    args: dict[str, Any], message: str
) -> None:
    """Fail closed before transforming malformed apply-model arguments."""

    with pytest.raises(ValueError, match=message):
        make_spatial_view_model_args(
            args=args,
            layout=_layout(_left_right_views(), input_batch_size=1),
        )


def test_spatial_args_preserve_batch_metadata_references_and_source_args() -> None:
    """Fix view ordering and all batch-aligned Comfy metadata transformations."""

    x = torch.arange(2 * 1 * 4 * 8, dtype=torch.float32).reshape((2, 1, 4, 8))
    timestep = torch.tensor([0.5, 0.75])
    reference = torch.arange(1 * 2 * 4 * 8, dtype=torch.float32).reshape((1, 2, 4, 8))
    untouched = object()
    existing_namespace = {"existing": untouched}
    conditioning: dict[str, Any] = {
        "c_concat": x.clone(),
        "c_crossattn": torch.tensor([[[1.0]], [[2.0]]]),
        "ref_latents": [reference],
        "transformer_options": {
            "cond_or_uncond": [0, 1],
            "uuids": ("positive", "negative"),
            "sigmas": torch.tensor([9.0, 8.0]),
            "untouched": untouched,
            "simple_syrup": existing_namespace,
        },
    }
    args: dict[str, Any] = {
        "input": x,
        "timestep": timestep,
        "cond_or_uncond": [0, 1],
        "c": conditioning,
        "passthrough": untouched,
    }

    layout = _layout(_left_right_views(), input_batch_size=2)
    transformed = make_spatial_view_model_args(
        args=args,
        layout=layout,
    )

    expected_x = torch.cat((x[..., :4], x[..., 4:]), dim=0)
    assert transformed is not args
    assert torch.equal(transformed["input"], expected_x)
    assert torch.equal(transformed["timestep"], torch.tensor([0.5, 0.75, 0.5, 0.75]))
    assert transformed["cond_or_uncond"] == [0, 1, 0, 1]
    assert transformed["passthrough"] is untouched

    transformed_conditioning = transformed["c"]
    assert torch.equal(transformed_conditioning["c_concat"], expected_x)
    assert torch.equal(
        transformed_conditioning["c_crossattn"],
        torch.tensor([[[1.0]], [[2.0]], [[1.0]], [[2.0]]]),
    )
    references = transformed_conditioning["ref_latents"]
    assert isinstance(references, list)
    assert torch.equal(references[0], torch.cat((reference,) * 4, dim=0))
    options = transformed_conditioning["transformer_options"]
    assert options["cond_or_uncond"] == [0, 1, 0, 1]
    assert options["uuids"] == ("positive", "negative", "positive", "negative")
    assert torch.equal(options["sigmas"], transformed["timestep"])
    assert options["untouched"] is untouched
    assert options is not conditioning["transformer_options"]
    assert options["simple_syrup"] is not existing_namespace
    assert options["simple_syrup"]["existing"] is untouched
    assert options["simple_syrup"]["spatial_batch_layout"] is layout

    assert args["input"] is x
    assert args["timestep"] is timestep
    assert args["c"] is conditioning
    assert conditioning["transformer_options"]["sigmas"].tolist() == [9.0, 8.0]
    assert conditioning["transformer_options"]["simple_syrup"] is existing_namespace
    assert "spatial_batch_layout" not in existing_namespace


@pytest.mark.parametrize(
    ("transformer_options", "message"),
    [
        ({"simple_syrup": "invalid"}, "simple_syrup must be a dictionary"),
        ("invalid", "transformer_options must be a dictionary"),
    ],
)
def test_spatial_args_reject_malformed_transformer_metadata(
    transformer_options: object,
    message: str,
) -> None:
    """Fail rather than overwrite a non-mapping metadata boundary."""

    with pytest.raises(TypeError, match=message):
        make_spatial_view_model_args(
            args={
                "input": torch.zeros((1, 1, 4, 8)),
                "timestep": torch.ones((1,)),
                "c": {"transformer_options": transformer_options},
            },
            layout=_layout(_left_right_views(), input_batch_size=1),
        )


def test_spatial_args_crop_only_final_axes_for_bcdhw() -> None:
    """Preserve singleton depth while batching equal spatial view rectangles."""

    x = torch.arange(1 * 2 * 1 * 4 * 8, dtype=torch.float32).reshape((1, 2, 1, 4, 8))

    transformed = make_spatial_view_model_args(
        args={"input": x, "timestep": torch.tensor([1.0]), "c": {}},
        layout=_layout(_left_right_views(), input_batch_size=1),
    )

    assert transformed["input"].shape == (2, 2, 1, 4, 4)
    assert torch.equal(transformed["input"][0:1], x[..., :4])
    assert torch.equal(transformed["input"][1:2], x[..., 4:])
    assert torch.equal(transformed["timestep"], torch.tensor([1.0, 1.0]))
    assert transformed["c"]["transformer_options"]["simple_syrup"][
        "spatial_batch_layout"
    ] == _layout(_left_right_views(), input_batch_size=1)


def test_full_view_projects_complete_multi_item_batch_without_change() -> None:
    """Project a full canvas through the same canonical spatial-call contract."""

    x = torch.arange(64, dtype=torch.float32).reshape((2, 1, 4, 8))
    timestep = torch.tensor([0.5, 0.75])
    layout = _layout(
        (SpatialView(SpatialViewKind.FULL, 0, 0, 8, 4, 8, 4),),
        input_batch_size=2,
    )

    transformed = make_spatial_view_model_args(
        args={
            "input": x,
            "timestep": timestep,
            "cond_or_uncond": [0, 1],
            "c": {"transformer_options": {"uuids": ("positive", "negative")}},
        },
        layout=layout,
    )

    assert torch.equal(transformed["input"], x)
    assert torch.equal(transformed["timestep"], timestep)
    assert transformed["cond_or_uncond"] == [0, 1]
    options = transformed["c"]["transformer_options"]
    assert options["uuids"] == ("positive", "negative")
    assert options["simple_syrup"]["spatial_batch_layout"] is layout


def _layout(
    views: tuple[SpatialView, ...],
    *,
    input_batch_size: int,
) -> SpatialBatchLayout:
    """Return one valid test layout for an eight-by-four canvas."""

    return SpatialBatchLayout(
        canvas_width=8,
        canvas_height=4,
        views=views,
        input_batch_size=input_batch_size,
    )
