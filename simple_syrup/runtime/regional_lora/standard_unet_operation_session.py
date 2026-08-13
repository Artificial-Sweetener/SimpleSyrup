# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve standard-UNet regional operation invocations inside one model call."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...domain.regional_activation_geometry import (
    RegionalActivationGeometry,
    RegionalActivationLayout,
    RegionalTemporalOwnership,
)
from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_lora_plan import RegionalLoraPlan
from ...domain.regional_mask_bank import RegionalMaskBank
from ...domain.spatial_views import SpatialBatchLayout
from ...masking.regional_activation_mask_projection import (
    REGIONAL_ACTIVATION_MASK_PROJECTOR,
)
from ...masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from ..attention_coupling.unet_attn2_execution import UnetAttn2Execution
from ..regional_attention_model_call_values import uniform_model_call_sigma
from ..regional_lora_schedule_resolution import RegionalLoraScheduleSession
from ..spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)
from .activation_batch_alignment import (
    REGIONAL_ACTIVATION_BATCH_ALIGNMENT_RESOLVER,
)
from .convolution_execution_plan import RegionalConvolutionExecutionPlan
from .convolution_rank_geometry import REGIONAL_CONVOLUTION_RANK_GEOMETRY_RESOLVER
from .linear_execution_plan import RegionalLinearExecutionPlan
from .operation_invocation import (
    RegionalOperationExecutionPlan,
    RegionalOperationInvocation,
)
from .operation_mask_resolution import REGIONAL_OPERATION_MASK_RESOLVER
from .standard_unet_packed_operation_masks import (
    STANDARD_UNET_PACKED_OPERATION_MASK_RESOLVER,
)
from .target_binding import BoundRegionalLoraSpatialCapability


@dataclass(slots=True)
class _ActiveStandardUnetOperationCall:
    """Retain exact call authorities and optional pending compact attn2 scope."""

    contexts: BatchedRegionalAttentionContexts
    transformer_options: dict[str, object]
    schedule_multipliers: tuple[float, ...]
    packed_execution: UnetAttn2Execution | None = None


@dataclass(frozen=True, slots=True)
class _StandardUnetScheduleSlot:
    """Bind one sampling schedule tensor to its stateful canonical cursor."""

    sample_sigmas: torch.Tensor
    maximum_sigma: float
    session: RegionalLoraScheduleSession
    last_sigma: float | None


class StandardUnetRegionalOperationSession:
    """Own live geometry, packed-branch, mask, and schedule invocation resolution."""

    def __init__(
        self,
        plan: RegionalLoraPlan,
        mask_bank: RegionalMaskBank,
        module_roles: Mapping[str, BoundRegionalLoraSpatialCapability],
    ) -> None:
        """Retain immutable composition, mask, and installed consumer-role evidence."""

        if not isinstance(plan, RegionalLoraPlan) or not plan.adapters:
            raise ValueError("Standard UNet operation session requires adapters.")
        if not isinstance(mask_bank, RegionalMaskBank):
            raise TypeError("Standard UNet operation session requires a mask bank.")
        if not isinstance(module_roles, Mapping) or not module_roles:
            raise ValueError("Standard UNet operation session requires module roles.")
        roles = dict(module_roles)
        if any(not isinstance(path, str) or not path for path in roles):
            raise ValueError("Standard UNet operation module paths must be nonempty.")
        if any(
            role
            not in (
                BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
                BoundRegionalLoraSpatialCapability.PACKED_IMAGE_TOKENS,
                BoundRegionalLoraSpatialCapability.PACKED_CONTEXT_TOKENS,
                BoundRegionalLoraSpatialCapability.DIRECT,
            )
            for role in roles.values()
        ):
            raise ValueError("Standard UNet operation module role is unsupported.")
        self._plan = plan
        self._mask_bank = mask_bank
        self._module_roles = roles
        self._active: ContextVar[_ActiveStandardUnetOperationCall | None] = ContextVar(
            "simple_syrup_standard_unet_regional_operation_call",
            default=None,
        )
        self._schedule: ContextVar[_StandardUnetScheduleSlot | None] = ContextVar(
            "simple_syrup_standard_unet_regional_operation_schedule",
            default=None,
        )

    @contextmanager
    def activate(
        self,
        contexts: BatchedRegionalAttentionContexts,
        transformer_options: dict[str, object],
    ) -> Iterator[None]:
        """Publish one exact denoiser call and its resolved schedule multipliers."""

        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Standard UNet operation call requires aligned contexts.")
        if not isinstance(transformer_options, dict):
            raise TypeError(
                "Standard UNet operation call requires transformer options."
            )
        resolution = self._resolve_schedule(transformer_options)
        active = _ActiveStandardUnetOperationCall(
            contexts,
            transformer_options,
            resolution,
        )
        token = self._active.set(active)
        try:
            yield
        finally:
            active.packed_execution = None
            self._active.reset(token)

    def begin_packed(self, execution: UnetAttn2Execution) -> None:
        """Publish one compact attn2 execution until its paired output callback."""

        active = self._require_active()
        if active.packed_execution is not None:
            raise ValueError("Standard UNet operation call already has packed state.")
        if not isinstance(execution, UnetAttn2Execution):
            raise TypeError("Standard UNet packed state requires attn2 execution.")
        active.packed_execution = execution

    def end_packed(self, execution: UnetAttn2Execution) -> None:
        """Clear only the compact execution opened by the input callback."""

        active = self._require_active()
        if active.packed_execution is not execution:
            raise ValueError("Standard UNet packed output does not match input state.")
        active.packed_execution = None

    def resolve(
        self,
        module_path: str,
        plan: RegionalOperationExecutionPlan,
        inputs: torch.Tensor,
    ) -> RegionalOperationInvocation | None:
        """Resolve one installed operation from its declared consumer role."""

        active = self._require_active()
        role = self._module_roles.get(module_path)
        if role is None:
            return None
        schedules = tuple(
            active.schedule_multipliers[use.composition_index] for use in plan.uses
        )
        if role is BoundRegionalLoraSpatialCapability.PACKED_IMAGE_TOKENS:
            execution = self._require_packed(active)
            if not isinstance(plan, RegionalLinearExecutionPlan):
                raise TypeError("Packed image role requires a Linear plan.")
            masks = STANDARD_UNET_PACKED_OPERATION_MASK_RESOLVER.resolve_image_tokens(
                execution,
                uses=plan.uses,
                inputs=inputs,
            )
        elif role is BoundRegionalLoraSpatialCapability.PACKED_CONTEXT_TOKENS:
            execution = self._require_packed(active)
            if not isinstance(plan, RegionalLinearExecutionPlan):
                raise TypeError("Packed context role requires a Linear plan.")
            masks = STANDARD_UNET_PACKED_OPERATION_MASK_RESOLVER.resolve_context_tokens(
                execution,
                uses=plan.uses,
                inputs=inputs,
            )
        else:
            if active.packed_execution is not None:
                raise ValueError("Ordinary regional operation ran inside packed attn2.")
            geometry = self._ordinary_geometry(
                role,
                plan=plan,
                inputs=inputs,
                active=active,
            )
            spatial = REGIONAL_ACTIVATION_MASK_PROJECTOR.project(
                bank=self._mask_bank,
                geometry=geometry,
                form=RegionalMaskForm.CONDITIONING,
                mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            masks = REGIONAL_OPERATION_MASK_RESOLVER.resolve(
                spatial,
                contexts=active.contexts,
                uses=plan.uses,
            )
        return RegionalOperationInvocation(masks, schedules)

    def clear(self, model: object, unpatch_all: bool) -> None:
        """Release retained task-local sampling schedule state on model detach."""

        del model, unpatch_all
        self._schedule.set(None)

    def _ordinary_geometry(
        self,
        role: BoundRegionalLoraSpatialCapability,
        *,
        plan: RegionalOperationExecutionPlan,
        inputs: torch.Tensor,
        active: _ActiveStandardUnetOperationCall,
    ) -> RegionalActivationGeometry:
        """Resolve exact ordinary image-token or direct-convolution geometry."""

        layout = _spatial_layout(active.transformer_options)
        alignment = REGIONAL_ACTIVATION_BATCH_ALIGNMENT_RESOLVER.resolve(
            active.contexts,
            spatial_layout=layout,
        )
        if role is BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS:
            if not isinstance(plan, RegionalLinearExecutionPlan) or inputs.ndim != 3:
                raise ValueError("Spatial-token role requires B/S/C Linear inputs.")
            activation_shape = _activation_shape(active.transformer_options)
            if int(inputs.shape[0]) != activation_shape[0] or int(inputs.shape[1]) != (
                activation_shape[2] * activation_shape[3]
            ):
                raise ValueError("Spatial-token inputs must match live activation H/W.")
            return RegionalActivationGeometry(
                RegionalActivationLayout.CONSUMER_SPATIALIZED,
                tuple(inputs.shape),
                2,
                activation_shape[2],
                activation_shape[3],
                alignment,
            )
        if role is not BoundRegionalLoraSpatialCapability.DIRECT or not isinstance(
            plan,
            RegionalConvolutionExecutionPlan,
        ):
            raise ValueError("Standard UNet ordinary operation role is inconsistent.")
        use = plan.uses[0]
        spatial = REGIONAL_CONVOLUTION_RANK_GEOMETRY_RESOLVER.resolve(
            tuple(int(value) for value in inputs.shape[2:]),
            use,
        )
        rank_channels = int(use.preparation.down.shape[0]) * use.parameters.groups
        layouts = {
            1: RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
            2: RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            3: RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
        }
        return RegionalActivationGeometry(
            layouts[use.parameters.dimension],
            (int(inputs.shape[0]), rank_channels, *spatial),
            1,
            1 if use.parameters.dimension == 1 else spatial[-2],
            spatial[-1],
            alignment,
            temporal_axis=2 if use.parameters.dimension == 3 else None,
            temporal_ownership=(
                RegionalTemporalOwnership.REPEAT_SPATIAL_MASK
                if use.parameters.dimension == 3
                else RegionalTemporalOwnership.NONE
            ),
        )

    def _resolve_schedule(
        self,
        transformer_options: dict[str, object],
    ) -> tuple[float, ...]:
        """Advance the existing schedule authority for one denoiser sigma."""

        sample_sigmas = _floating_tensor(
            transformer_options.get("sample_sigmas"),
            "sample_sigmas",
        )
        current_sigmas = _floating_tensor(
            transformer_options.get("sigmas"),
            "sigmas",
        )
        current_sigma = uniform_model_call_sigma(current_sigmas)
        slot = self._schedule.get()
        maximum = (
            _maximum_finite_sigma(sample_sigmas)
            if slot is None or slot.sample_sigmas is not sample_sigmas
            else slot.maximum_sigma
        )
        if (
            slot is None
            or slot.sample_sigmas is not sample_sigmas
            or (slot.last_sigma is not None and current_sigma > slot.last_sigma)
        ):
            slot = _StandardUnetScheduleSlot(
                sample_sigmas,
                maximum,
                RegionalLoraScheduleSession(
                    self._plan.adapters,
                    maximum_sigma=maximum,
                ),
                None,
            )
        resolution = slot.session.resolve(current_sigma)
        self._schedule.set(
            _StandardUnetScheduleSlot(
                sample_sigmas,
                slot.maximum_sigma,
                slot.session,
                current_sigma,
            )
        )
        return resolution.schedule_multipliers

    def _require_active(self) -> _ActiveStandardUnetOperationCall:
        """Return the current call or reject execution outside its owner."""

        active = self._active.get()
        if active is None:
            raise RuntimeError("Standard UNet regional operation ran outside a call.")
        return active

    @staticmethod
    def _require_packed(
        active: _ActiveStandardUnetOperationCall,
    ) -> UnetAttn2Execution:
        """Return the pending compact attn2 authority."""

        if active.packed_execution is None:
            raise RuntimeError("Packed regional operation ran outside attn2 scope.")
        return active.packed_execution


def _activation_shape(options: dict[str, object]) -> tuple[int, int, int, int]:
    """Narrow Comfy's live spatial-transformer BCHW metadata."""

    value = options.get("activations_shape")
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise TypeError("Standard UNet activations_shape must be a BCHW sequence.")
    shape = tuple(value)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in shape
    ):
        raise ValueError("Standard UNet activation dimensions must be positive.")
    return shape[0], shape[1], shape[2], shape[3]


def _spatial_layout(options: dict[str, object]) -> SpatialBatchLayout | None:
    """Return the optional authoritative full/tiled/Contextual call layout."""

    namespace = options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
    if namespace is None:
        return None
    if not isinstance(namespace, dict):
        raise TypeError("Standard UNet SimpleSyrup namespace must be a dictionary.")
    layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
    if layout is not None and not isinstance(layout, SpatialBatchLayout):
        raise TypeError("Standard UNet spatial layout has an invalid type.")
    return layout


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    """Require one nonempty floating Comfy sigma tensor."""

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.numel() < 1
    ):
        raise TypeError(
            f"Standard UNet regional LoRA {name} must be a floating tensor."
        )
    return value


def _maximum_finite_sigma(sample_sigmas: torch.Tensor) -> float:
    """Validate a new sampling schedule on CPU and return its maximum."""

    values = tuple(
        float(value) for value in sample_sigmas.flatten().detach().cpu().tolist()
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Standard UNet regional LoRA sample_sigmas must be finite.")
    return max(values)
