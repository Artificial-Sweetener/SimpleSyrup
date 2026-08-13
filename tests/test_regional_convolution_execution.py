# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact generic regional convolution execution against references."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
)
from simple_syrup.runtime.regional_lora.convolution_execution import (
    RegionalConvolutionExecutor,
)
from simple_syrup.runtime.regional_lora.convolution_execution_plan import (
    RegionalConvolutionExecutionPlan,
    RegionalConvolutionParameters,
    RegionalConvolutionTargetUse,
)
from simple_syrup.runtime.regional_lora.convolution_preparation import (
    RegionalConvolutionPreparation,
)
from simple_syrup.runtime.regional_lora.operation_mask_resolution import (
    RegionalOperationMaskBatch,
)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_direct_convolution_dimensions_match_rank_space_reference(
    dimension: int,
) -> None:
    """Match exact Conv1d/2d/3d down-mask-up math with spatial kernels."""

    fixture = _fixture(dimension=dimension, kernel_size=3, padding=1)
    mask = torch.linspace(0.0, 1.0, fixture.rank_element_count).reshape(
        1,
        *fixture.rank_spatial,
    )

    result = fixture.execute(mask, schedule=0.6)

    rank = _conv(
        fixture.inputs,
        fixture.down,
        stride=fixture.parameters.stride,
        padding=fixture.parameters.padding,
        dilation=fixture.parameters.dilation,
        groups=fixture.parameters.groups,
    )
    expected = fixture.original(fixture.inputs) + _conv(
        rank * mask.unsqueeze(1) * 0.8 * 0.6,
        fixture.up,
        stride=(1,) * dimension,
        padding=(0,) * dimension,
        dilation=(1,) * dimension,
        groups=1,
    )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_pointwise_convolution_all_one_matches_global_delta() -> None:
    """Match ordinary complete-image convolution LoRA at full coverage."""

    fixture = _fixture(dimension=2, kernel_size=1, padding=0)

    result = fixture.execute(torch.ones((1, *fixture.rank_spatial)))

    expected = fixture.original(fixture.inputs) + _conv(
        _conv(
            fixture.inputs,
            fixture.down,
            stride=fixture.parameters.stride,
            padding=fixture.parameters.padding,
            dilation=fixture.parameters.dilation,
            groups=1,
        )
        * 0.8,
        fixture.up,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_locon_middle_applies_mask_after_spatial_middle() -> None:
    """Project pointwise down through middle kernel before regional masking."""

    fixture = _fixture(dimension=2, kernel_size=3, padding=1, use_middle=True)
    mask = torch.linspace(0.2, 0.9, fixture.rank_element_count).reshape(
        1,
        *fixture.rank_spatial,
    )

    result = fixture.execute(mask)

    pointwise = _conv(
        fixture.inputs,
        fixture.down,
        stride=fixture.parameters.stride,
        padding=fixture.parameters.padding,
        dilation=fixture.parameters.dilation,
        groups=1,
    )
    assert fixture.middle is not None
    rank = _conv(
        pointwise,
        fixture.middle,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    expected = fixture.original(fixture.inputs) + _conv(
        rank * mask.unsqueeze(1) * 0.8,
        fixture.up,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_stride_changes_rank_mask_and_output_resolution_exactly() -> None:
    """Require masks at down-convolution output resolution, not input resolution."""

    fixture = _fixture(dimension=2, kernel_size=3, padding=1, stride=2)
    assert fixture.inputs.shape[-2:] != fixture.rank_spatial

    result = fixture.execute(torch.ones((1, *fixture.rank_spatial)))

    assert tuple(result.shape[-2:]) == fixture.rank_spatial
    with pytest.raises(ValueError, match="exact rank activation shape"):
        fixture.execute(torch.ones((1, *fixture.inputs.shape[-2:])))


def test_zero_mask_and_schedule_skip_preparation_but_call_original_once() -> None:
    """Avoid device preparation for inactive convolution adapters."""

    fixture = _fixture(dimension=2, kernel_size=3, padding=1)
    original = _CountedOriginal(fixture.original)

    result = fixture.execute(
        torch.zeros((1, *fixture.rank_spatial)),
        original=original,
    )

    assert torch.equal(result, fixture.original(fixture.inputs))
    assert original.calls == 1
    assert fixture.use.preparation._prepared == {}


def test_multiple_adapters_preserve_declared_addition_order() -> None:
    """Add distinct convolution targets in declared schedule order."""

    fixture = _fixture(dimension=2, kernel_size=3, padding=1)
    second_down = fixture.down * -0.5
    second_up = fixture.up * 0.25
    second = RegionalConvolutionTargetUse(
        1,
        1,
        RegionalLoraBranch.POSITIVE,
        (id(second_down), id(second_up), None),
        RegionalConvolutionPreparation(second_down, None, second_up),
        fixture.parameters,
        0.4,
    )
    plan = RegionalConvolutionExecutionPlan((fixture.use, second))
    masks = torch.stack(
        (
            torch.full(fixture.rank_spatial, 0.25),
            torch.full(fixture.rank_spatial, 0.75),
        )
    )

    result = fixture.execute(masks, schedules=(0.5, 0.75), plan=plan)

    expected = fixture.original(fixture.inputs)
    for down, up, mask, scale in (
        (fixture.down, fixture.up, masks[0], 0.8 * 0.5),
        (second_down, second_up, masks[1], 0.4 * 0.75),
    ):
        rank = _conv(
            fixture.inputs,
            down,
            stride=fixture.parameters.stride,
            padding=fixture.parameters.padding,
            dilation=fixture.parameters.dilation,
            groups=1,
        )
        expected = expected + _conv(
            rank * mask.unsqueeze(0).unsqueeze(0) * scale,
            up,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_grouped_convolution_matches_grouped_merged_weight_reference() -> None:
    """Repeat rank bases per group so all-one execution equals merged weight math."""

    original = nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=True)
    inputs = torch.linspace(-1.0, 1.0, 2 * 4 * 5 * 5).reshape(2, 4, 5, 5)
    down = torch.linspace(-0.2, 0.3, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
    up = torch.linspace(-0.4, 0.25, 6 * 2).reshape(6, 2, 1, 1)
    parameters = RegionalConvolutionParameters(
        2,
        (1, 1),
        (1, 1),
        (1, 1),
        2,
        4,
        6,
        (3, 3),
    )
    use = RegionalConvolutionTargetUse(
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        (id(down), id(up), None),
        RegionalConvolutionPreparation(down, None, up),
        parameters,
        0.75,
    )
    plan = RegionalConvolutionExecutionPlan((use,))
    masks = _mask_batch(torch.ones((1, 5, 5)), dimension=2, input_batch=2, features=4)

    result = RegionalConvolutionExecutor().execute(
        original,
        inputs,
        plan=plan,
        masks=masks,
        schedule_strengths=(1.0,),
    )

    adapter = LoRAAdapter(
        {"up", "down"},
        (up, down, None, None, None, None),
    )
    patched = adapter.calculate_weight(
        original.weight.detach().clone(),
        "diffusion_model.grouped.weight",
        0.75,
        1.0,
        None,
        lambda value: value,
        intermediate_dtype=torch.float32,
    )
    expected = functional.conv2d(
        inputs,
        patched,
        original.bias,
        stride=1,
        padding=1,
        groups=2,
    )
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_view_major_tiled_batch_keeps_each_tile_mask_on_its_rows() -> None:
    """Apply distinct view masks over existing view/chunk/latent batch order."""

    fixture = _fixture(dimension=2, kernel_size=1, padding=0)
    fixture.inputs = fixture.inputs.repeat(2, 1, 1, 1)
    fixture.rank_spatial = tuple(int(value) for value in fixture.inputs.shape[2:])
    view_values = torch.cat(
        (
            torch.full((2, *fixture.rank_spatial), 0.25),
            torch.full((2, *fixture.rank_spatial), 0.75),
        ),
        dim=0,
    )
    spatial_masks = RegionalActivationMaskBatch(
        view_values.unsqueeze(0).unsqueeze(2),
        RegionalActivationGeometry(
            RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            (4, 2, *fixture.rank_spatial),
            1,
            fixture.rank_spatial[0],
            fixture.rank_spatial[1],
            RegionalActivationBatchAlignment(2, 1, _two_view_layout()),
        ),
    )
    masks = RegionalOperationMaskBatch(
        spatial_masks.multipliers,
        spatial_masks.geometry,
        (0,),
    )

    result = RegionalConvolutionExecutor().execute(
        fixture.original,
        fixture.inputs,
        plan=fixture.plan,
        masks=masks,
        schedule_strengths=(1.0,),
    )

    rank = _conv(
        fixture.inputs,
        fixture.down,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    expected = fixture.original(fixture.inputs) + _conv(
        rank * view_values.unsqueeze(1) * 0.8,
        fixture.up,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_matches_explicit_reference(dtype: torch.dtype) -> None:
    """Preserve full-rank convolution LoRA quality in FP16 and BF16."""

    fixture = _fixture(
        dimension=2,
        kernel_size=3,
        padding=1,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    mask = torch.linspace(
        0.0,
        1.0,
        fixture.rank_element_count,
        device="cuda",
        dtype=dtype,
    ).reshape(1, *fixture.rank_spatial)

    result = fixture.execute(mask)

    rank = _conv(
        fixture.inputs,
        fixture.down.to("cuda", dtype),
        stride=fixture.parameters.stride,
        padding=fixture.parameters.padding,
        dilation=fixture.parameters.dilation,
        groups=1,
    )
    expected = fixture.original(fixture.inputs) + _conv(
        rank * mask.unsqueeze(1) * 0.8,
        fixture.up.to("cuda", dtype),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.02)


class _Fixture:
    """Retain one deterministic dimensional convolution execution case."""

    def __init__(
        self,
        *,
        dimension: int,
        kernel_size: int,
        padding: int,
        stride: int,
        use_middle: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Create exact original and low-rank convolution tensors."""

        convolution = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dimension - 1]
        spatial = (7,) * dimension
        self.original = convolution(
            2,
            3,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.inputs = torch.linspace(
            -1.0,
            1.0,
            2 * 2 * (7**dimension),
            device=device,
            dtype=dtype,
        ).reshape(2, 2, *spatial)
        rank = 2
        down_kernel = (1,) * dimension if use_middle else (kernel_size,) * dimension
        self.down = torch.linspace(
            -0.3,
            0.4,
            rank * 2 * (down_kernel[0] ** dimension),
        ).reshape(rank, 2, *down_kernel)
        self.middle = (
            torch.linspace(-0.2, 0.25, rank * rank * (kernel_size**dimension)).reshape(
                rank,
                rank,
                *((kernel_size,) * dimension),
            )
            if use_middle
            else None
        )
        self.up = torch.linspace(-0.4, 0.35, 3 * rank).reshape(
            3,
            rank,
            *((1,) * dimension),
        )
        self.parameters = RegionalConvolutionParameters(
            dimension,
            (stride,) * dimension,
            (padding,) * dimension,
            (1,) * dimension,
            1,
            2,
            3,
            (kernel_size,) * dimension,
        )
        self.use = RegionalConvolutionTargetUse(
            0,
            0,
            RegionalLoraBranch.POSITIVE,
            (
                id(self.down),
                id(self.up),
                None if self.middle is None else id(self.middle),
            ),
            RegionalConvolutionPreparation(self.down, self.middle, self.up),
            self.parameters,
            0.8,
        )
        self.plan = RegionalConvolutionExecutionPlan((self.use,))
        self.rank_spatial = _reference_rank_spatial(self.inputs, self.use)
        self.rank_element_count = int(torch.tensor(self.rank_spatial).prod())

    def execute(
        self,
        masks: torch.Tensor,
        *,
        schedule: float = 1.0,
        schedules: tuple[float, ...] | None = None,
        plan: RegionalConvolutionExecutionPlan | None = None,
        original: object | None = None,
    ) -> torch.Tensor:
        """Execute this fixture with explicitly shaped rank masks."""

        selected_plan = plan or self.plan
        mask_batch = _mask_batch(
            masks.to(self.inputs),
            dimension=self.parameters.dimension,
            input_batch=int(self.inputs.shape[0]),
            features=2,
        )
        selected_original = self.original if original is None else original
        assert callable(selected_original)
        return RegionalConvolutionExecutor().execute(
            selected_original,
            self.inputs,
            plan=selected_plan,
            masks=mask_batch,
            schedule_strengths=schedules or (schedule,),
        )


class _CountedOriginal:
    """Count original convolution calls while preserving output behavior."""

    def __init__(self, original: nn.Module) -> None:
        """Retain the original module and zero calls."""

        self._original = original
        self.calls = 0

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Count and forward one exact original call."""

        self.calls += 1
        result = self._original(inputs)
        assert isinstance(result, torch.Tensor)
        return result


def _fixture(
    *,
    dimension: int,
    kernel_size: int,
    padding: int,
    stride: int = 1,
    use_middle: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> _Fixture:
    """Return one deterministic convolution execution fixture."""

    return _Fixture(
        dimension=dimension,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        use_middle=use_middle,
        device=device or torch.device("cpu"),
        dtype=dtype,
    )


def _mask_batch(
    values: torch.Tensor,
    *,
    dimension: int,
    input_batch: int,
    features: int,
) -> RegionalOperationMaskBatch:
    """Wrap region/spatial masks in direct convolution geometry."""

    if values.ndim == dimension:
        values = values.unsqueeze(0)
    regions = int(values.shape[0])
    spatial_shape = tuple(int(value) for value in values.shape[1:])
    invocation = (input_batch, features, *spatial_shape)
    layout = {
        1: RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
        2: RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
        3: RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
    }[dimension]
    geometry = RegionalActivationGeometry(
        layout,
        invocation,
        1,
        1 if dimension == 1 else spatial_shape[-2],
        spatial_shape[-1],
        RegionalActivationBatchAlignment(input_batch, 1),
        temporal_axis=2 if dimension == 3 else None,
        temporal_ownership=(
            RegionalTemporalOwnership.REPEAT_SPATIAL_MASK
            if dimension == 3
            else RegionalTemporalOwnership.NONE
        ),
    )
    expanded = values.reshape(regions, 1, 1, *spatial_shape).expand(
        -1,
        input_batch,
        -1,
        *spatial_shape,
    )
    spatial_masks = RegionalActivationMaskBatch(expanded, geometry)
    return RegionalOperationMaskBatch(
        spatial_masks.multipliers,
        spatial_masks.geometry,
        tuple(range(regions)),
    )


def _reference_rank_spatial(
    inputs: torch.Tensor,
    use: RegionalConvolutionTargetUse,
) -> tuple[int, ...]:
    """Observe the exact explicit down/middle rank activation shape."""

    rank = _conv(
        inputs,
        use.preparation.down.to(inputs),
        stride=use.parameters.stride,
        padding=use.parameters.padding,
        dilation=use.parameters.dilation,
        groups=use.parameters.groups,
    )
    if use.preparation.middle is not None:
        rank = _conv(
            rank,
            use.preparation.middle.to(inputs),
            stride=(1,) * use.parameters.dimension,
            padding=(0,) * use.parameters.dimension,
            dilation=(1,) * use.parameters.dimension,
            groups=1,
        )
    return tuple(int(value) for value in rank.shape[2:])


def _conv(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> torch.Tensor:
    """Dispatch one explicit dimensional reference convolution."""

    function = (functional.conv1d, functional.conv2d, functional.conv3d)[
        inputs.ndim - 3
    ]
    return function(inputs, weight, None, stride, padding, dilation, groups)


def _two_view_layout() -> SpatialBatchLayout:
    """Return two equal view-major tiles over a four-entry active batch."""

    return SpatialBatchLayout(
        14,
        7,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 7, 7, 7, 7),
            SpatialView(SpatialViewKind.TILE, 7, 0, 7, 7, 7, 7),
        ),
        input_batch_size=2,
    )
