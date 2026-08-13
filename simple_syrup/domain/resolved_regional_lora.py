# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define model-neutral resolved regional ordinary-LoRA operation contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .regional_lora_plan import RegionalLoraAdapterPlan


class ResolvedLoraOperationClass(StrEnum):
    """Classify normalized low-rank tensor organization before module binding."""

    MATRIX_PAIR = "matrix_pair"
    CONVOLUTION_1D = "convolution_1d"
    CONVOLUTION_2D = "convolution_2d"
    CONVOLUTION_3D = "convolution_3d"
    UNSUPPORTED = "unsupported"


class RegionalLoraGeometryClass(StrEnum):
    """Declare the activation geometry an execution contract requires."""

    TARGET_OPERATION = "target_operation"
    DIRECT_CONVOLUTION_1D = "direct_convolution_1d"
    DIRECT_CONVOLUTION_2D = "direct_convolution_2d"
    DIRECT_CONVOLUTION_3D = "direct_convolution_3d"
    UNSUPPORTED = "unsupported"


class RegionalLoraExecutionContract(StrEnum):
    """Declare exact rank-activation execution or explicit rejection."""

    MATRIX_OR_RESHAPED_CONVOLUTION = "matrix_or_reshaped_convolution"
    DIRECT_CONVOLUTION = "direct_convolution"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class RegionalLoraTensorShape:
    """Retain one immutable positive tensor shape without tensor ownership."""

    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require a nonempty tuple of strictly positive integer dimensions."""

        if not isinstance(self.dimensions, tuple) or not self.dimensions:
            raise ValueError("Regional LoRA tensor shape must be a nonempty tuple.")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.dimensions
        ):
            raise ValueError(
                "Regional LoRA tensor shape dimensions must be positive integers."
            )

    @property
    def rank(self) -> int:
        """Return the tensor dimensionality."""

        return len(self.dimensions)


@dataclass(frozen=True, slots=True)
class ResolvedRegionalLoraTarget:
    """Retain one model-relative parameter path and optional Comfy tensor slice."""

    model_target: str
    parameter_name: str
    offset: tuple[int, ...] | None

    def __post_init__(self) -> None:
        """Require an explicit module target, parameter, and valid optional slice."""

        if not isinstance(self.model_target, str) or not self.model_target.strip():
            raise ValueError("Resolved regional LoRA model target must be nonempty.")
        if not isinstance(self.parameter_name, str) or not self.parameter_name.strip():
            raise ValueError("Resolved regional LoRA parameter name must be nonempty.")
        if self.offset is not None and (
            not isinstance(self.offset, tuple)
            or not self.offset
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in self.offset
            )
        ):
            raise ValueError(
                "Resolved regional LoRA target offset must contain "
                "non-negative integers."
            )

    @property
    def parameter_path(self) -> str:
        """Return the complete model-relative parameter path."""

        return f"{self.model_target}.{self.parameter_name}"


@dataclass(frozen=True, slots=True)
class ResolvedRegionalLoraOperation:
    """Describe one supported operation or one explicit target rejection."""

    adapter: RegionalLoraAdapterPlan
    target_index: int
    target: ResolvedRegionalLoraTarget
    normalized_operation_type: str
    operation_class: ResolvedLoraOperationClass
    down_shape: RegionalLoraTensorShape | None
    up_shape: RegionalLoraTensorShape | None
    middle_shape: RegionalLoraTensorShape | None
    reshape_shape: RegionalLoraTensorShape | None
    rank: int | None
    intrinsic_scale: float | None
    required_geometry: RegionalLoraGeometryClass
    execution_contract: RegionalLoraExecutionContract
    rejection_reason: str | None

    def __post_init__(self) -> None:
        """Reject mutable, incomplete, or mathematically inconsistent metadata."""

        if not isinstance(self.adapter, RegionalLoraAdapterPlan):
            raise TypeError("Resolved regional LoRA operation requires an adapter.")
        _require_non_negative_index(self.target_index, name="target_index")
        if not isinstance(self.target, ResolvedRegionalLoraTarget):
            raise TypeError("Resolved regional LoRA operation requires a target.")
        if (
            not isinstance(self.normalized_operation_type, str)
            or not self.normalized_operation_type.strip()
        ):
            raise ValueError(
                "Normalized regional LoRA operation type must be nonempty."
            )
        if not isinstance(self.operation_class, ResolvedLoraOperationClass):
            raise TypeError("Resolved regional LoRA operation class is invalid.")
        if not isinstance(self.required_geometry, RegionalLoraGeometryClass):
            raise TypeError("Resolved regional LoRA geometry class is invalid.")
        if not isinstance(self.execution_contract, RegionalLoraExecutionContract):
            raise TypeError("Resolved regional LoRA execution contract is invalid.")
        if self.execution_contract is RegionalLoraExecutionContract.UNSUPPORTED:
            self._validate_rejection()
            return
        self._validate_supported()

    def _validate_rejection(self) -> None:
        """Require one reason and no misleading supported-operation metadata."""

        if self.operation_class is not ResolvedLoraOperationClass.UNSUPPORTED:
            raise ValueError(
                "Rejected regional LoRA operation class must be unsupported."
            )
        if self.required_geometry is not RegionalLoraGeometryClass.UNSUPPORTED:
            raise ValueError("Rejected regional LoRA geometry must be unsupported.")
        if (
            not isinstance(self.rejection_reason, str)
            or not self.rejection_reason.strip()
        ):
            raise ValueError("Rejected regional LoRA operation requires a reason.")
        if any(
            value is not None
            for value in (
                self.down_shape,
                self.up_shape,
                self.middle_shape,
                self.reshape_shape,
                self.rank,
                self.intrinsic_scale,
            )
        ):
            raise ValueError(
                "Rejected regional LoRA operation cannot claim executable metadata."
            )

    def _validate_supported(self) -> None:
        """Require complete shape, scale, geometry, and execution consistency."""

        if self.operation_class is ResolvedLoraOperationClass.UNSUPPORTED:
            raise ValueError("Supported regional LoRA operation cannot be unsupported.")
        if self.required_geometry is RegionalLoraGeometryClass.UNSUPPORTED:
            raise ValueError("Supported regional LoRA geometry cannot be unsupported.")
        if self.rejection_reason is not None:
            raise ValueError(
                "Supported regional LoRA operation cannot have a rejection."
            )
        if self.down_shape is None or self.up_shape is None:
            raise ValueError(
                "Supported regional LoRA operation requires down/up shapes."
            )
        if self.down_shape.rank < 2 or self.up_shape.rank < 2:
            raise ValueError(
                "Regional LoRA down/up shapes need at least two dimensions."
            )
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ValueError("Supported regional LoRA rank must be a positive integer.")
        if not isinstance(self.intrinsic_scale, float) or not math.isfinite(
            self.intrinsic_scale
        ):
            raise ValueError("Regional LoRA intrinsic scale must be finite.")
        if self.down_shape.dimensions[0] != self.rank:
            raise ValueError("Regional LoRA down shape must begin with its rank.")
        if (
            len(self.up_shape.dimensions) < 2
            or self.up_shape.dimensions[1] != self.rank
        ):
            raise ValueError(
                "Regional LoRA up shape must contain its rank at index one."
            )
        if self.middle_shape is not None:
            if self.middle_shape.rank < 2:
                raise ValueError(
                    "Regional LoRA middle shape needs at least two dimensions."
                )
            if (
                self.middle_shape.dimensions[0] != self.rank
                or self.middle_shape.dimensions[1] != self.rank
            ):
                raise ValueError(
                    "Regional LoRA middle shape must preserve rank channels."
                )
        self._validate_operation_shape_contract()

    def _validate_operation_shape_contract(self) -> None:
        """Match operation, tensor dimensionality, geometry, and execution policy."""

        if self.operation_class is ResolvedLoraOperationClass.MATRIX_PAIR:
            if self.down_shape is None or self.up_shape is None:
                raise AssertionError(
                    "Supported shape validation requires down/up shapes."
                )
            if self.down_shape.rank != 2 or self.up_shape.rank != 2:
                raise ValueError("Matrix-pair regional LoRA tensors must be rank two.")
            if self.middle_shape is not None:
                raise ValueError(
                    "Matrix-pair regional LoRA cannot contain middle weights."
                )
            if self.required_geometry is not RegionalLoraGeometryClass.TARGET_OPERATION:
                raise ValueError("Matrix-pair regional LoRA requires target geometry.")
            if (
                self.execution_contract
                is not RegionalLoraExecutionContract.MATRIX_OR_RESHAPED_CONVOLUTION
            ):
                raise ValueError("Matrix-pair regional LoRA execution is inconsistent.")
            return
        spatial_rank = {
            ResolvedLoraOperationClass.CONVOLUTION_1D: 3,
            ResolvedLoraOperationClass.CONVOLUTION_2D: 4,
            ResolvedLoraOperationClass.CONVOLUTION_3D: 5,
        }[self.operation_class]
        expected_geometry = {
            3: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_1D,
            4: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
            5: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_3D,
        }[spatial_rank]
        shapes = tuple(
            shape
            for shape in (self.down_shape, self.up_shape, self.middle_shape)
            if shape is not None
        )
        if any(shape.rank not in (2, spatial_rank) for shape in shapes):
            raise ValueError("Convolutional regional LoRA tensor rank is inconsistent.")
        if not any(shape.rank == spatial_rank for shape in shapes):
            raise ValueError("Convolutional regional LoRA needs one spatial tensor.")
        if self.required_geometry is not expected_geometry:
            raise ValueError("Convolutional regional LoRA geometry is inconsistent.")
        if (
            self.execution_contract
            is not RegionalLoraExecutionContract.DIRECT_CONVOLUTION
        ):
            raise ValueError("Convolutional regional LoRA execution is inconsistent.")

    @property
    def authored_strength(self) -> float:
        """Return the authored model strength from the authoritative plan owner."""

        return self.adapter.model_strength

    @property
    def supported(self) -> bool:
        """Report whether this target has an executable ordinary-LoRA contract."""

        return self.execution_contract is not RegionalLoraExecutionContract.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class ResolvedRegionalLoraOperationSet:
    """Retain canonical adapter/target order across supported and rejected entries."""

    entries: tuple[ResolvedRegionalLoraOperation, ...]

    def __post_init__(self) -> None:
        """Require immutable entries in contiguous per-adapter target order."""

        if not isinstance(self.entries, tuple):
            raise TypeError("Resolved regional LoRA entries must be a tuple.")
        if any(
            not isinstance(entry, ResolvedRegionalLoraOperation)
            for entry in self.entries
        ):
            raise TypeError("Resolved regional LoRA set contains an invalid entry.")
        observed = tuple(
            (entry.adapter.composition_index, entry.target_index)
            for entry in self.entries
        )
        if observed != tuple(sorted(observed)):
            raise ValueError(
                "Resolved regional LoRA entries must retain declared order."
            )
        indices_by_adapter: dict[int, list[int]] = {}
        for composition_index, target_index in observed:
            indices_by_adapter.setdefault(composition_index, []).append(target_index)
        if any(
            target_indices != list(range(len(target_indices)))
            for target_indices in indices_by_adapter.values()
        ):
            raise ValueError(
                "Resolved regional LoRA target indices must be contiguous per adapter."
            )

    @property
    def supported(self) -> tuple[ResolvedRegionalLoraOperation, ...]:
        """Return supported entries without changing canonical relative order."""

        return tuple(entry for entry in self.entries if entry.supported)

    @property
    def rejected(self) -> tuple[ResolvedRegionalLoraOperation, ...]:
        """Return rejected entries without changing canonical relative order."""

        return tuple(entry for entry in self.entries if not entry.supported)


def _require_non_negative_index(value: object, *, name: str) -> None:
    """Require one non-negative non-boolean integer index."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Resolved regional LoRA {name} must be non-negative.")
