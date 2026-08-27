# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project captured attention observations through one bounded evidence grid."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional

from ..domain.attention_geometry import factor_spatial_geometry
from ..domain.attention_region_maps import CapturedAttentionMap
from .attention_spatial_projection import ATTENTION_SPATIAL_PROJECTION_SERVICE

MAXIMUM_ANALYSIS_EDGE = 192


class AttentionObservationProjectionService:
    """Own geometry-safe observation projection and final-size restoration."""

    def stack(
        self,
        observations: tuple[CapturedAttentionMap, ...],
        values: Callable[[CapturedAttentionMap], torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Project observation values into one memory-bounded spatial stack."""

        analysis_height, analysis_width = self.analysis_geometry(height, width)
        return torch.stack(
            tuple(
                self._project_observation(
                    observation,
                    values(observation),
                    analysis_height,
                    analysis_width,
                )
                for observation in observations
            )
        )

    def restore(
        self,
        values: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Restore one analysis-grid map to the public output dimensions."""

        if tuple(values.shape) == (height, width):
            return values
        return functional.interpolate(
            values.reshape(1, 1, int(values.shape[0]), int(values.shape[1])),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    def analysis_geometry(self, height: int, width: int) -> tuple[int, int]:
        """Return a bounded grid that preserves the requested aspect ratio."""

        if max(height, width) <= MAXIMUM_ANALYSIS_EDGE:
            return height, width
        scale = MAXIMUM_ANALYSIS_EDGE / float(max(height, width))
        return max(1, round(height * scale)), max(1, round(width * scale))

    def _project_observation(
        self,
        observation: CapturedAttentionMap,
        values: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Project one captured vector using its authoritative source geometry."""

        source_height = observation.spatial_height
        source_width = observation.spatial_width
        if source_height is None or source_width is None:
            source_height, source_width = factor_spatial_geometry(
                int(values.numel()), target_aspect=width / height
            )
        source = values.float().reshape(source_height, source_width)
        projected = ATTENTION_SPATIAL_PROJECTION_SERVICE.project(
            source,
            observation.spatial_transforms,
        )
        return functional.interpolate(
            projected.reshape(
                1,
                1,
                int(projected.shape[0]),
                int(projected.shape[1]),
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]


ATTENTION_OBSERVATION_PROJECTION_SERVICE = AttentionObservationProjectionService()
