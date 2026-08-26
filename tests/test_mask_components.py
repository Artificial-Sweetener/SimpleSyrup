# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize shared connected-component extraction."""

from __future__ import annotations

import torch

from simple_syrup.masking.mask_components import connected_mask_components


def test_connected_components_are_eight_connected_and_spatially_ordered() -> None:
    """Join diagonal pixels and return components from top-left to bottom-right."""

    active = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, False, True],
        ]
    )

    components = connected_mask_components(active)

    assert tuple(component.bbox for component in components) == (
        (0, 0, 2, 2),
        (3, 2, 4, 3),
    )
    assert tuple(int(component.mask.count_nonzero()) for component in components) == (
        2,
        1,
    )


def test_connected_components_return_empty_for_an_empty_mask() -> None:
    """Return no components without inventing geometry."""

    assert connected_mask_components(torch.zeros(4, 5, dtype=torch.bool)) == ()
