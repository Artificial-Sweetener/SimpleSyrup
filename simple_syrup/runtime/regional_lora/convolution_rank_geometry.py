# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve exact regional convolution rank-space activation dimensions."""

from __future__ import annotations

from .convolution_execution_plan import RegionalConvolutionTargetUse


class RegionalConvolutionRankGeometryResolver:
    """Own ordinary down/optional-middle convolution output-size math."""

    def resolve(
        self,
        input_spatial: tuple[int, ...],
        use: RegionalConvolutionTargetUse,
    ) -> tuple[int, ...]:
        """Return exact rank-activation spatial dimensions before the up path."""

        if not isinstance(input_spatial, tuple) or not input_spatial:
            raise ValueError("Regional convolution input spatial shape is required.")
        if not isinstance(use, RegionalConvolutionTargetUse):
            raise TypeError("Regional convolution rank geometry requires a target use.")
        parameters = use.parameters
        if len(input_spatial) != parameters.dimension or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in input_spatial
        ):
            raise ValueError("Regional convolution input spatial shape is invalid.")
        down = use.preparation.down
        if down.ndim == 2 or use.preparation.middle is None:
            down_kernel = parameters.kernel_size
        else:
            down_kernel = tuple(int(value) for value in down.shape[2:])
        spatial = tuple(
            _output_size(size, kernel, stride, padding, dilation)
            for size, kernel, stride, padding, dilation in zip(
                input_spatial,
                down_kernel,
                parameters.stride,
                parameters.padding,
                parameters.dilation,
                strict=True,
            )
        )
        middle = use.preparation.middle
        if middle is not None:
            middle_kernel = tuple(int(value) for value in middle.shape[2:])
            spatial = tuple(
                _output_size(size, kernel, 1, 0, 1)
                for size, kernel in zip(spatial, middle_kernel, strict=True)
            )
        if any(size < 1 for size in spatial):
            raise ValueError("Regional convolution rank activation would be empty.")
        return spatial


def _output_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Return Torch's ordinary convolution output size for one axis."""

    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


REGIONAL_CONVOLUTION_RANK_GEOMETRY_RESOLVER = RegionalConvolutionRankGeometryResolver()
