"""Verify regional activity pruning and exact fast-path classification."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import torch

from simple_syrup.masking.regional_mask_activation import (
    RegionalCoverageClass,
    RegionalMaskActivation,
    RegionalMaskActivationClassifier,
)


def test_all_zero_masks_classify_as_all_base() -> None:
    """Prune every zero-coverage region and admit the base-only fast path."""

    activation = RegionalMaskActivationClassifier().classify(torch.zeros((3, 2, 4)))

    assert activation == RegionalMaskActivation(
        coverage_class=RegionalCoverageClass.ALL_BASE,
        active_region_indices=(),
        single_region_index=None,
    )


def test_one_full_region_classifies_as_all_single_region() -> None:
    """Admit one region only when it covers the complete query grid."""

    masks = torch.zeros((3, 2, 4))
    masks[1] = 1.0

    activation = RegionalMaskActivationClassifier().classify(masks)

    assert activation.coverage_class is RegionalCoverageClass.ALL_SINGLE_REGION
    assert activation.active_region_indices == (1,)
    assert activation.single_region_index == 1


def test_partial_single_region_remains_mixed_with_base() -> None:
    """Retain the base branch when one region leaves any query position uncovered."""

    masks = torch.zeros((2, 2, 4))
    masks[1, :, :2] = 1.0

    activation = RegionalMaskActivationClassifier().classify(masks)

    assert activation.coverage_class is RegionalCoverageClass.MIXED
    assert activation.active_region_indices == (1,)
    assert activation.single_region_index is None


def test_multiple_active_regions_remain_ordered_after_zero_pruning() -> None:
    """Preserve authored identity while excluding interleaved zero regions."""

    masks = torch.zeros((5, 2, 4))
    masks[1, :, :2] = 0.25
    masks[4, :, 2:] = 0.75

    activation = RegionalMaskActivationClassifier().classify(masks)

    assert activation.coverage_class is RegionalCoverageClass.MIXED
    assert activation.active_region_indices == (1, 4)


def test_tolerance_prunes_noise_and_admits_near_unit_coverage() -> None:
    """Use one explicit tolerance for zero pruning and complete coverage."""

    masks = torch.full((2, 2, 2), 5e-5)
    masks[1] = 1.0 - 5e-5

    activation = RegionalMaskActivationClassifier().classify(
        masks,
        tolerance=1e-4,
    )

    assert activation.coverage_class is RegionalCoverageClass.ALL_SINGLE_REGION
    assert activation.active_region_indices == (1,)
    assert activation.single_region_index == 1


def test_activation_result_is_frozen() -> None:
    """Prevent consumers from changing an admitted fast-path classification."""

    activation = RegionalMaskActivationClassifier().classify(torch.zeros((1, 1, 1)))
    attribute = "single_region_index"

    with pytest.raises(FrozenInstanceError):
        setattr(activation, attribute, 0)


@pytest.mark.parametrize(
    ("masks", "error", "message"),
    [
        (cast(Any, "invalid"), TypeError, "must be a torch.Tensor"),
        (torch.ones((2, 2)), ValueError, "must use BHW layout"),
        (torch.ones((0, 2, 2)), ValueError, "at least one region"),
        (torch.ones((1, 0, 2)), ValueError, "grids must be non-empty"),
        (
            torch.ones((1, 2, 2), dtype=torch.int64),
            TypeError,
            "floating-point dtype",
        ),
        (
            torch.tensor([[[float("inf")]]]),
            ValueError,
            "contain finite values",
        ),
        (torch.tensor([[[-0.1]]]), ValueError, r"stay within \[0, 1\]"),
        (torch.tensor([[[1.1]]]), ValueError, r"stay within \[0, 1\]"),
    ],
)
def test_classifier_rejects_invalid_projected_masks(
    masks: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed before classifying malformed projected coverage."""

    with pytest.raises(error, match=message):
        RegionalMaskActivationClassifier().classify(masks)


@pytest.mark.parametrize("tolerance", [-0.1, 0.5, 1.0])
def test_classifier_rejects_invalid_tolerance(tolerance: float) -> None:
    """Keep zero and complete-coverage thresholds non-overlapping."""

    with pytest.raises(ValueError, match="at least 0 and below 0.5"):
        RegionalMaskActivationClassifier().classify(
            torch.ones((1, 1, 1)),
            tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "activation",
    [
        RegionalMaskActivation(RegionalCoverageClass.ALL_BASE, (), None),
        RegionalMaskActivation(RegionalCoverageClass.ALL_SINGLE_REGION, (2,), 2),
        RegionalMaskActivation(RegionalCoverageClass.MIXED, (1,), None),
    ],
)
def test_valid_activation_value_invariants(
    activation: RegionalMaskActivation,
) -> None:
    """Accept each internally consistent immutable classification shape."""

    assert isinstance(activation.coverage_class, RegionalCoverageClass)


@pytest.mark.parametrize(
    ("coverage_class", "indices", "single_index", "message"),
    [
        (RegionalCoverageClass.ALL_BASE, (0,), None, "cannot contain"),
        (RegionalCoverageClass.ALL_SINGLE_REGION, (), None, "requires its one"),
        (RegionalCoverageClass.ALL_SINGLE_REGION, (1,), 0, "requires its one"),
        (RegionalCoverageClass.MIXED, (), None, "requires an active"),
        (RegionalCoverageClass.MIXED, (1,), 1, "cannot select"),
        (RegionalCoverageClass.MIXED, (2, 1), None, "unique and ordered"),
        (RegionalCoverageClass.MIXED, (1, 1), None, "unique and ordered"),
        (RegionalCoverageClass.MIXED, (-1,), None, "non-negative"),
    ],
)
def test_activation_value_rejects_inconsistent_state(
    coverage_class: RegionalCoverageClass,
    indices: tuple[int, ...],
    single_index: int | None,
    message: str,
) -> None:
    """Prevent construction of contradictory fast-path metadata."""

    with pytest.raises(ValueError, match=message):
        RegionalMaskActivation(coverage_class, indices, single_index)
