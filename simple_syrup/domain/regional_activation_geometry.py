# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define model-neutral regional activation geometry and batch alignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .spatial_views import SpatialBatchLayout


class RegionalActivationLayout(StrEnum):
    """Classify the explicit spatial organization of one adapter activation."""

    DIRECT_CONVOLUTION_1D = "direct_convolution_1d"
    DIRECT_CONVOLUTION_2D = "direct_convolution_2d"
    DIRECT_CONVOLUTION_3D = "direct_convolution_3d"
    FLATTENED_SPATIAL_TOKENS = "flattened_spatial_tokens"
    CONSUMER_SPATIALIZED = "consumer_spatialized"


class RegionalTemporalOwnership(StrEnum):
    """Declare how a two-dimensional authored mask owns temporal activations."""

    NONE = "none"
    REPEAT_SPATIAL_MASK = "repeat_spatial_mask"


@dataclass(frozen=True, slots=True)
class RegionalActivationBatchAlignment:
    """Retain CFG, latent-batch, and optional view-major alignment evidence."""

    latent_batch_size: int
    chunk_count: int
    spatial_layout: SpatialBatchLayout | None = None

    def __post_init__(self) -> None:
        """Require positive counts and one layout over the complete base batch."""

        _positive_integer(self.latent_batch_size, name="latent_batch_size")
        _positive_integer(self.chunk_count, name="chunk_count")
        if self.spatial_layout is not None:
            if not isinstance(self.spatial_layout, SpatialBatchLayout):
                raise TypeError(
                    "Regional activation spatial_layout must be a SpatialBatchLayout."
                )
            if self.spatial_layout.input_batch_size != self.base_batch_size:
                raise ValueError(
                    "Regional activation spatial layout input batch must match "
                    "CFG chunks times latent batch size."
                )

    @property
    def base_batch_size(self) -> int:
        """Return the model batch before spatial-view expansion."""

        return self.latent_batch_size * self.chunk_count

    @property
    def invocation_batch_size(self) -> int:
        """Return the active model batch after optional view expansion."""

        if self.spatial_layout is None:
            return self.base_batch_size
        return self.spatial_layout.expanded_batch_size


@dataclass(frozen=True, slots=True)
class RegionalActivationGeometry:
    """Describe one exact rank activation and its authored-mask correspondence."""

    layout: RegionalActivationLayout
    invocation_shape: tuple[int, ...]
    feature_axis: int
    spatial_height: int
    spatial_width: int
    batch_alignment: RegionalActivationBatchAlignment
    temporal_axis: int | None = None
    temporal_ownership: RegionalTemporalOwnership = RegionalTemporalOwnership.NONE

    def __post_init__(self) -> None:
        """Reject ambiguous axes, batches, tokens, and temporal ownership."""

        if not isinstance(self.layout, RegionalActivationLayout):
            raise TypeError("Regional activation layout has an invalid type.")
        if not isinstance(self.invocation_shape, tuple) or not self.invocation_shape:
            raise ValueError("Regional activation shape must be a nonempty tuple.")
        for dimension in self.invocation_shape:
            _positive_integer(dimension, name="shape dimension")
        if not isinstance(self.batch_alignment, RegionalActivationBatchAlignment):
            raise TypeError("Regional activation requires batch alignment evidence.")
        _positive_integer(self.spatial_height, name="spatial_height")
        _positive_integer(self.spatial_width, name="spatial_width")
        if self.invocation_shape[0] != self.batch_alignment.invocation_batch_size:
            raise ValueError(
                "Regional activation leading batch must match its alignment."
            )
        spatial_layout = self.batch_alignment.spatial_layout
        if spatial_layout is not None:
            mismatched_views = tuple(
                index
                for index, view in enumerate(spatial_layout.views)
                if (view.model_height, view.model_width)
                != (self.spatial_height, self.spatial_width)
            )
            if mismatched_views:
                raise ValueError(
                    "Regional activation H/W must match every published spatial "
                    f"view; mismatched view indices {mismatched_views}."
                )
        rank = len(self.invocation_shape)
        _axis(self.feature_axis, rank=rank, name="feature_axis")
        if self.feature_axis == 0:
            raise ValueError(
                "Regional activation feature axis cannot be the batch axis."
            )
        if not isinstance(self.temporal_ownership, RegionalTemporalOwnership):
            raise TypeError(
                "Regional activation temporal ownership has an invalid type."
            )
        self._validate_layout()

    def _validate_layout(self) -> None:
        """Match the declared layout to its exact conventional tensor shape."""

        batch = self.batch_alignment.invocation_batch_size
        features = self.invocation_shape[self.feature_axis]
        if self.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_1D:
            self._require_shape((batch, features, self.spatial_width), feature_axis=1)
            if self.spatial_height != 1:
                raise ValueError("Direct Conv1d regional geometry requires height one.")
            self._require_no_temporal_axis()
            return
        if self.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_2D:
            self._require_shape(
                (batch, features, self.spatial_height, self.spatial_width),
                feature_axis=1,
            )
            self._require_no_temporal_axis()
            return
        if self.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_3D:
            if self.feature_axis != 1 or len(self.invocation_shape) != 5:
                raise ValueError("Direct Conv3d regional geometry requires B/C/D/H/W.")
            if self.temporal_axis != 2:
                raise ValueError(
                    "Direct Conv3d regional geometry requires temporal axis 2."
                )
            if self.invocation_shape[3:] != (
                self.spatial_height,
                self.spatial_width,
            ):
                raise ValueError("Direct Conv3d regional H/W must match its tensor.")
            if (
                self.temporal_ownership
                is not RegionalTemporalOwnership.REPEAT_SPATIAL_MASK
            ):
                raise ValueError(
                    "Direct Conv3d regional geometry requires explicit repeated "
                    "spatial-mask temporal ownership."
                )
            return
        if self.layout in (
            RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
            RegionalActivationLayout.CONSUMER_SPATIALIZED,
        ):
            self._require_shape(
                (
                    batch,
                    self.spatial_height * self.spatial_width,
                    features,
                ),
                feature_axis=2,
            )
            self._require_no_temporal_axis()
            return
        raise AssertionError(f"Unhandled regional activation layout: {self.layout}")

    def _require_shape(
        self,
        expected: tuple[int, ...],
        *,
        feature_axis: int,
    ) -> None:
        """Require one conventional shape and feature-axis location."""

        if self.feature_axis != feature_axis or self.invocation_shape != expected:
            raise ValueError(
                f"{self.layout.value} regional geometry expected shape {expected} "
                f"with feature axis {feature_axis}; observed "
                f"{self.invocation_shape} and axis {self.feature_axis}."
            )

    def _require_no_temporal_axis(self) -> None:
        """Reject temporal claims from non-temporal image activation layouts."""

        if self.temporal_axis is not None:
            raise ValueError(
                "Non-temporal regional geometry cannot declare a temporal axis."
            )
        if self.temporal_ownership is not RegionalTemporalOwnership.NONE:
            raise ValueError(
                "Non-temporal regional geometry cannot claim temporal ownership."
            )

    @property
    def temporal_size(self) -> int | None:
        """Return the explicit temporal size when this activation owns one."""

        if self.temporal_axis is None:
            return None
        return self.invocation_shape[self.temporal_axis]

    def broadcast_mask_shape(self, region_count: int) -> tuple[int, ...]:
        """Return the exact region-major multiplier shape for this activation."""

        _positive_integer(region_count, name="region_count")
        shape = list(self.invocation_shape)
        shape[self.feature_axis] = 1
        return (region_count, *shape)


def _positive_integer(value: object, *, name: str) -> None:
    """Require one strictly positive non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Regional activation {name} must be an integer.")
    if value < 1:
        raise ValueError(f"Regional activation {name} must be positive.")


def _axis(value: object, *, rank: int, name: str) -> None:
    """Require one non-negative axis inside the invocation rank."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Regional activation {name} must be an integer.")
    if not 0 <= value < rank:
        raise ValueError(f"Regional activation {name} is outside the tensor rank.")
