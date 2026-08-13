# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify spatial model calls fail closed before model execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.spatial_model_arguments import (
    make_spatial_view_model_args,
    validate_spatial_model_args,
)


def test_valid_spatial_model_args_pass_complete_contract_validation() -> None:
    """Accept an exact view-major expansion with the authoritative layout."""

    source, transformed, layout = _valid_model_args()

    validate_spatial_model_args(
        source_args=source,
        transformed_args=transformed,
        layout=layout,
    )


@pytest.mark.parametrize(
    ("key", "invalid_value", "message"),
    [
        ("input", torch.zeros((3, 1, 4, 4)), "Spatial model input batch must be 4"),
        ("timestep", torch.zeros((3,)), "Spatial model timestep batch must be 4"),
    ],
)
def test_validation_rejects_incomplete_expanded_tensor_batches(
    key: str,
    invalid_value: torch.Tensor,
    message: str,
) -> None:
    """Require one tensor row for every view and source-batch combination."""

    source, transformed, layout = _valid_model_args()
    transformed[key] = invalid_value

    with pytest.raises(ValueError, match=message):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


def test_validation_rejects_missing_simple_syrup_metadata() -> None:
    """Require spatial metadata to be published at the model-call boundary."""

    source, transformed, layout = _valid_model_args()
    _transformed_options(transformed).pop("simple_syrup")

    with pytest.raises(TypeError, match="require SimpleSyrup metadata"):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


def test_validation_rejects_non_authoritative_layout_instance() -> None:
    """Reject an equal layout that is not the instance governing this call."""

    source, transformed, layout = _valid_model_args()
    replacement = _layout()
    _transformed_options(transformed)["simple_syrup"] = {
        "spatial_batch_layout": replacement
    }

    with pytest.raises(ValueError, match="layout does not match the model call"):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


@pytest.mark.parametrize(
    ("metadata_owner", "key", "invalid_value", "message"),
    [
        ("top", "cond_or_uncond", [0, 0, 1, 1], "Top-level cond_or_uncond"),
        (
            "transformer",
            "cond_or_uncond",
            [0, 0, 1, 1],
            "Transformer cond_or_uncond",
        ),
        (
            "transformer",
            "uuids",
            ("positive", "positive", "negative", "negative"),
            "Transformer UUIDs",
        ),
    ],
)
def test_validation_rejects_source_major_metadata_permutations(
    metadata_owner: str,
    key: str,
    invalid_value: object,
    message: str,
) -> None:
    """Reject source-major metadata that would address the wrong spatial view."""

    source, transformed, layout = _valid_model_args()
    owner = (
        transformed if metadata_owner == "top" else _transformed_options(transformed)
    )
    owner[key] = invalid_value

    with pytest.raises(ValueError, match=rf"{message} is not in view-major order"):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


def test_source_major_permutation_reports_complete_layout_diagnostic() -> None:
    """Explain the exact layout contract when ordering alone is corrupted."""

    source, transformed, layout = _valid_model_args()
    transformed["cond_or_uncond"] = [0, 0, 1, 1]

    diagnostic = (
        "Top-level cond_or_uncond is not in view-major order for 2 spatial views "
        "over input batch 2; expected the source sequence repeated once per view."
    )
    with pytest.raises(ValueError) as captured:
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )

    assert str(captured.value) == diagnostic


def test_validation_rejects_transformer_sigma_batch_mismatch() -> None:
    """Require sigma metadata to address the complete expanded model batch."""

    source, transformed, layout = _valid_model_args()
    _transformed_options(transformed)["sigmas"] = torch.ones((3,))

    with pytest.raises(ValueError, match="Transformer sigmas batch must be 4"):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


def test_validation_rejects_transformer_sigma_timestep_misalignment() -> None:
    """Require sigma metadata to match the transformed timestep values exactly."""

    source, transformed, layout = _valid_model_args()
    _transformed_options(transformed)["sigmas"] = torch.zeros((4,))

    with pytest.raises(ValueError, match="sigmas must align with model timesteps"):
        validate_spatial_model_args(
            source_args=source,
            transformed_args=transformed,
            layout=layout,
        )


def _valid_model_args() -> tuple[
    dict[str, Any],
    dict[str, Any],
    SpatialBatchLayout,
]:
    """Return source and transformed arguments satisfying the spatial contract."""

    source: dict[str, Any] = {
        "input": torch.arange(64, dtype=torch.float32).reshape((2, 1, 4, 8)),
        "timestep": torch.tensor([0.5, 0.75]),
        "cond_or_uncond": [0, 1],
        "c": {
            "transformer_options": {
                "cond_or_uncond": [0, 1],
                "uuids": ("positive", "negative"),
                "sigmas": torch.tensor([0.5, 0.75]),
            }
        },
    }
    layout = _layout()
    transformed = make_spatial_view_model_args(args=source, layout=layout)
    return source, transformed, layout


def _transformed_options(transformed: dict[str, Any]) -> dict[str, Any]:
    """Return typed transformer options from transformed model arguments."""

    conditioning = transformed["c"]
    assert isinstance(conditioning, dict)
    options = conditioning["transformer_options"]
    assert isinstance(options, dict)
    return options


def _layout() -> SpatialBatchLayout:
    """Return two ordered tile views over a two-item source batch."""

    return SpatialBatchLayout(
        canvas_width=8,
        canvas_height=4,
        views=(
            SpatialView(SpatialViewKind.TILE, 0, 0, 4, 4, 4, 4),
            SpatialView(SpatialViewKind.TILE, 4, 0, 4, 4, 4, 4),
        ),
        input_batch_size=2,
    )
