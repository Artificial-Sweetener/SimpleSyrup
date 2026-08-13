# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provide regional activation geometry from explicit operation invocations."""

from __future__ import annotations

from typing import Protocol

import torch

from ...domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)


class RegionalActivationGeometryProvider(Protocol):
    """Resolve one exact regional activation without inferring hidden axes."""

    def resolve(
        self,
        activation: torch.Tensor,
        *,
        batch_alignment: RegionalActivationBatchAlignment,
    ) -> RegionalActivationGeometry:
        """Return validated geometry for one observed activation tensor."""

        ...


class DirectConvolutionGeometryProvider:
    """Describe channel-first Conv1d/2d/3d rank activations."""

    def __init__(
        self,
        *,
        temporal_ownership: RegionalTemporalOwnership = (
            RegionalTemporalOwnership.NONE
        ),
    ) -> None:
        """Retain only an explicit temporal ownership policy."""

        if not isinstance(temporal_ownership, RegionalTemporalOwnership):
            raise TypeError(
                "Direct convolution temporal ownership has an invalid type."
            )
        self._temporal_ownership = temporal_ownership

    def resolve(
        self,
        activation: torch.Tensor,
        *,
        batch_alignment: RegionalActivationBatchAlignment,
    ) -> RegionalActivationGeometry:
        """Describe one exact channel-first convolution result by tensor rank."""

        _validate_activation(activation, batch_alignment=batch_alignment)
        if activation.ndim == 3:
            return RegionalActivationGeometry(
                RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
                tuple(activation.shape),
                1,
                1,
                int(activation.shape[2]),
                batch_alignment,
            )
        if activation.ndim == 4:
            return RegionalActivationGeometry(
                RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
                tuple(activation.shape),
                1,
                int(activation.shape[2]),
                int(activation.shape[3]),
                batch_alignment,
            )
        if activation.ndim == 5:
            return RegionalActivationGeometry(
                RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
                tuple(activation.shape),
                1,
                int(activation.shape[3]),
                int(activation.shape[4]),
                batch_alignment,
                temporal_axis=2,
                temporal_ownership=self._temporal_ownership,
            )
        raise ValueError(
            "Direct regional convolution activation must use B/C/L, B/C/H/W, "
            "or B/C/D/H/W layout."
        )


class SpatialTokenGeometryProvider:
    """Describe flattened spatial tokens using caller-published height and width."""

    def __init__(
        self,
        *,
        spatial_height: int,
        spatial_width: int,
        consumer_spatialized: bool = False,
    ) -> None:
        """Retain the explicit producer/consumer spatial grid."""

        if type(spatial_height) is not int or spatial_height < 1:
            raise ValueError("Spatial token height must be a positive integer.")
        if type(spatial_width) is not int or spatial_width < 1:
            raise ValueError("Spatial token width must be a positive integer.")
        if not isinstance(consumer_spatialized, bool):
            raise TypeError("Consumer-spatialized selection must be boolean.")
        self._height = spatial_height
        self._width = spatial_width
        self._layout = (
            RegionalActivationLayout.CONSUMER_SPATIALIZED
            if consumer_spatialized
            else RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS
        )

    def resolve(
        self,
        activation: torch.Tensor,
        *,
        batch_alignment: RegionalActivationBatchAlignment,
    ) -> RegionalActivationGeometry:
        """Match one B/S/C activation to its explicitly published grid."""

        _validate_activation(activation, batch_alignment=batch_alignment)
        if activation.ndim != 3:
            raise ValueError("Spatial-token regional activation must use B/S/C layout.")
        return RegionalActivationGeometry(
            self._layout,
            tuple(activation.shape),
            2,
            self._height,
            self._width,
            batch_alignment,
        )


def _validate_activation(
    activation: object,
    *,
    batch_alignment: RegionalActivationBatchAlignment,
) -> None:
    """Require one finite floating activation and typed alignment evidence."""

    if not isinstance(activation, torch.Tensor):
        raise TypeError("Regional activation must be a torch.Tensor.")
    if not activation.is_floating_point():
        raise TypeError("Regional activation must use a floating-point dtype.")
    if not isinstance(batch_alignment, RegionalActivationBatchAlignment):
        raise TypeError("Regional activation requires typed batch alignment.")
    if activation.ndim < 1 or int(activation.shape[0]) != (
        batch_alignment.invocation_batch_size
    ):
        raise ValueError("Regional activation batch must match its alignment.")
