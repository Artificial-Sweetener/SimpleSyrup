# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact canonical-mask projection into regional rank activations."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
    RegionalActivationMaskProjector,
)
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)


def test_full_token_projection_preserves_soft_overlap_and_uncovered_pixels() -> None:
    """Flatten exact soft full-canvas masks without normalizing overlap or gaps."""

    masks = torch.tensor(
        [
            [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
            [[0.0, 0.75, 0.0], [0.0, 0.75, 0.0]],
        ]
    )
    bank = _bank(masks)
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
        (2, 6, 4),
        2,
        2,
        3,
        RegionalActivationBatchAlignment(1, 2),
    )

    projected = _project(bank, geometry, RegionalMaskProjectionMode.SOFT)

    expected = masks.flatten(start_dim=1).unsqueeze(1).expand(-1, 2, -1).unsqueeze(-1)
    torch.testing.assert_close(projected.multipliers, expected)
    assert float(projected.multipliers[:, :, 1].sum()) == 2.5
    assert projected.multipliers[:, :, 2].eq(0.0).all()


def test_tiled_projection_uses_view_chunk_then_latent_order() -> None:
    """Repeat each cropped tile over the layout's authoritative base batch."""

    masks = torch.tensor([[[0.25, 0.25, 0.75, 0.75]]]).expand(1, 2, 4).clone()
    bank = _bank(masks)
    layout = SpatialBatchLayout(
        4,
        2,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 2, 2, 2),
        ),
        input_batch_size=4,
    )
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
        (8, 3, 2, 2),
        1,
        2,
        2,
        RegionalActivationBatchAlignment(2, 2, layout),
    )

    projected = _project(bank, geometry, RegionalMaskProjectionMode.HARD_PRESERVING)

    torch.testing.assert_close(
        projected.multipliers[0, :, 0, 0, 0],
        torch.tensor([0.25] * 4 + [0.75] * 4),
    )


def test_contextual_projection_resizes_complete_source_to_model_grid() -> None:
    """Use the existing contextual full-source view as the crop/resize authority."""

    masks = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) / 23.0
    bank = _bank(masks)
    layout = SpatialBatchLayout(
        6,
        4,
        (
            SpatialView(
                SpatialViewKind.CONTEXTUAL_GLOBAL,
                0,
                0,
                6,
                4,
                3,
                2,
            ),
        ),
        input_batch_size=1,
    )
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.CONSUMER_SPATIALIZED,
        (1, 6, 5),
        2,
        2,
        3,
        RegionalActivationBatchAlignment(1, 1, layout),
    )

    projected = _project(
        bank,
        geometry,
        RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
    )

    assert projected.multipliers.shape == (1, 1, 6, 1)
    assert projected.multipliers.min() >= 0.0
    assert projected.multipliers.max() <= 1.0


def test_conv3d_projection_repeats_spatial_ownership_over_declared_time() -> None:
    """Repeat one authored image mask only under explicit temporal ownership."""

    bank = _bank(torch.tensor([[[1.0, 0.0], [0.5, 0.25]]]))
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
        (1, 4, 3, 2, 2),
        1,
        2,
        2,
        RegionalActivationBatchAlignment(1, 1),
        temporal_axis=2,
        temporal_ownership=RegionalTemporalOwnership.REPEAT_SPATIAL_MASK,
    )

    projected = _project(bank, geometry, RegionalMaskProjectionMode.SOFT)

    assert projected.multipliers.shape == (1, 1, 1, 3, 2, 2)
    torch.testing.assert_close(
        projected.multipliers[:, :, :, 0],
        projected.multipliers[:, :, :, 2],
    )


def test_projection_does_not_mutate_canonical_masks() -> None:
    """Leave tensor identity, version, dtype, device, and contents unchanged."""

    masks = torch.rand((2, 3, 5))
    bank = _bank(masks)
    before = _fingerprint(bank)
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
        (1, 4, 2, 4),
        1,
        2,
        4,
        RegionalActivationBatchAlignment(1, 1),
    )

    _project(bank, geometry, RegionalMaskProjectionMode.CONTINUOUS_COVERAGE)

    assert _fingerprint(bank) == before


def _project(
    bank: RegionalMaskBank,
    geometry: RegionalActivationGeometry,
    mode: RegionalMaskProjectionMode,
) -> RegionalActivationMaskBatch:
    """Project one test case through the complete typed boundary."""

    return RegionalActivationMaskProjector().project(
        bank=bank,
        geometry=geometry,
        form=RegionalMaskForm.CONDITIONING,
        mode=mode,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _bank(masks: torch.Tensor) -> RegionalMaskBank:
    """Create independent planning and conditioning masks."""

    return RegionalMaskBank(
        masks.clone(),
        masks.clone(),
        int(masks.shape[2]),
        int(masks.shape[1]),
    )


def _fingerprint(bank: RegionalMaskBank) -> tuple[tuple[object, ...], ...]:
    """Capture source identity and state without copying its contents."""

    return tuple(
        (
            tensor.data_ptr(),
            tensor._version,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            float(tensor.sum()),
        )
        for tensor in (bank.planning_masks, bank.conditioning_masks)
    )
