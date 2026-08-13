# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Map declared scaling work to its immutable observed value."""

from __future__ import annotations

from .matrix_manifest import PerformanceWorkExpectation
from .matrix_measurement import ScalingWorkObservation


def expected_scaling_work(
    expectation: PerformanceWorkExpectation,
) -> ScalingWorkObservation:
    """Return the exact observation represented by one matrix declaration."""

    return ScalingWorkObservation(
        expectation.active_adapter_uses,
        expectation.active_target_count,
        expectation.target_use_count,
        expectation.deduplicated_target_group_count,
        expectation.compatible_projection_batch_count,
        expectation.deduplicated_target_uses,
    )
