# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify bounded comparison of independently materialized variant banks."""

from __future__ import annotations

import math

import pytest
import torch

from simple_syrup.runtime.regional_lora.standard_unet_variant_materialization import (
    StandardUnetMaterializedVariant,
    StandardUnetVariantParameter,
)
from tools.attention_coupling_benchmark.comfy_probe import (
    materialized_variant_comparison,
)

compare_materialized_variants = (
    materialized_variant_comparison.compare_materialized_variants
)


def _variant(
    region_index: int,
    *parameters: tuple[str, torch.Tensor],
) -> StandardUnetMaterializedVariant:
    """Build one canonical detached test bank."""

    return StandardUnetMaterializedVariant(
        region_index,
        tuple(
            StandardUnetVariantParameter(path, tensor.detach())
            for path, tensor in parameters
        ),
    )


def test_identical_independent_banks_have_equal_hashes_and_zero_error() -> None:
    """Treat equal values as exact without relying on tensor identity."""

    reference = _variant(
        2,
        ("block.bias", torch.tensor([1.0, -2.0], dtype=torch.float16)),
        ("block.weight", torch.arange(12, dtype=torch.float16).reshape(3, 4)),
    )
    candidate = _variant(
        2,
        ("block.bias", reference.parameters[0].tensor.clone()),
        ("block.weight", reference.parameters[1].tensor.clone()),
    )

    result = compare_materialized_variants(reference, candidate, chunk_elements=3)

    assert result.region_index == 2
    assert result.parameter_count == 2
    assert result.element_count == 14
    assert result.differing_element_count == 0
    assert result.max_absolute_error == 0.0
    assert result.mean_absolute_error == 0.0
    assert result.root_mean_squared_error == 0.0
    assert result.reference_sha256 == result.candidate_sha256
    assert result.exact is True
    assert result.max_error_parameter_path is None


def test_changed_element_reports_exact_aggregate_error() -> None:
    """Aggregate one changed value without hiding it behind an average."""

    reference = _variant(
        0,
        ("block.weight", torch.tensor([1.0, 2.0, 3.0, 4.0])),
    )
    candidate = _variant(
        0,
        ("block.weight", torch.tensor([1.0, 2.5, 3.0, 4.0])),
    )

    result = compare_materialized_variants(reference, candidate, chunk_elements=2)

    assert result.differing_element_count == 1
    assert result.max_absolute_error == 0.5
    assert result.mean_absolute_error == 0.125
    assert result.root_mean_squared_error == 0.25
    assert result.reference_sha256 != result.candidate_sha256
    assert result.exact is False
    assert result.max_error_parameter_path == "block.weight"


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        (
            _variant(0, ("a", torch.ones(2))),
            _variant(0, ("b", torch.ones(2))),
            "paths",
        ),
        (
            _variant(0, ("a", torch.ones(2))),
            _variant(0, ("a", torch.ones(3))),
            "shape",
        ),
        (
            _variant(0, ("a", torch.ones(2, dtype=torch.float16))),
            _variant(0, ("a", torch.ones(2, dtype=torch.float32))),
            "dtype",
        ),
    ],
)
def test_structural_mismatch_fails_closed(
    reference: StandardUnetMaterializedVariant,
    candidate: StandardUnetMaterializedVariant,
    message: str,
) -> None:
    """Reject banks whose semantic parameter structures do not align."""

    with pytest.raises(ValueError, match=message):
        compare_materialized_variants(reference, candidate)


def test_nonfinite_parameter_fails_closed() -> None:
    """Avoid publishing meaningless bounded-error metrics for nonfinite values."""

    reference = _variant(0, ("a", torch.tensor([1.0, math.inf])))
    candidate = _variant(0, ("a", torch.tensor([1.0, math.inf])))

    with pytest.raises(ValueError, match="finite"):
        compare_materialized_variants(reference, candidate)
