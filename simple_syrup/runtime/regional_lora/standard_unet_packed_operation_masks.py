# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve regional operation masks for compact standard-UNet attn2 branches."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
)
from ..attention_coupling.unet_attn2_execution import UnetAttn2Execution
from .operation_mask_resolution import (
    REGIONAL_OPERATION_BRANCH_GATE_RESOLVER,
    RegionalOperationMaskBatch,
    RegionalOperationMaskUse,
)


class StandardUnetPackedOperationMaskResolver:
    """Map regional uses onto exact compact attn2 image or context rows."""

    def resolve_image_tokens(
        self,
        execution: UnetAttn2Execution,
        *,
        uses: Sequence[RegionalOperationMaskUse],
        inputs: torch.Tensor,
    ) -> RegionalOperationMaskBatch:
        """Return query-grid masks for packed Q/output Linear calls."""

        self._validate(execution, uses=uses, inputs=inputs)
        if int(inputs.shape[1]) != execution.query_height * execution.query_width:
            raise ValueError("Packed image tokens must match attn2 query H/W.")
        use_masks = tuple(
            self._packed_use_mask(execution, use=use, inputs=inputs, spatial=True)
            for use in uses
        )
        geometry = RegionalActivationGeometry(
            RegionalActivationLayout.CONSUMER_SPATIALIZED,
            tuple(inputs.shape),
            2,
            execution.query_height,
            execution.query_width,
            RegionalActivationBatchAlignment(int(inputs.shape[0]), 1),
        )
        return RegionalOperationMaskBatch(
            torch.stack(use_masks),
            geometry,
            tuple(use.composition_index for use in uses),
        )

    def resolve_context_tokens(
        self,
        execution: UnetAttn2Execution,
        *,
        uses: Sequence[RegionalOperationMaskUse],
        inputs: torch.Tensor,
    ) -> RegionalOperationMaskBatch:
        """Return row gates broadcast over untouched native context tokens."""

        self._validate(execution, uses=uses, inputs=inputs)
        use_masks = tuple(
            self._packed_use_mask(execution, use=use, inputs=inputs, spatial=False)
            for use in uses
        )
        geometry = RegionalActivationGeometry(
            RegionalActivationLayout.BRANCH_TOKENS,
            tuple(inputs.shape),
            2,
            1,
            int(inputs.shape[1]),
            RegionalActivationBatchAlignment(int(inputs.shape[0]), 1),
        )
        return RegionalOperationMaskBatch(
            torch.stack(use_masks),
            geometry,
            tuple(use.composition_index for use in uses),
        )

    @staticmethod
    def _validate(
        execution: object,
        *,
        uses: Sequence[RegionalOperationMaskUse],
        inputs: object,
    ) -> None:
        """Require one exact packed B/S/C activation and canonical use order."""

        if not isinstance(execution, UnetAttn2Execution):
            raise TypeError("Packed operation masks require an attn2 execution.")
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
            raise ValueError("Packed operation inputs must use B/S/C layout.")
        if int(inputs.shape[0]) != execution.branches.packed_batch_size:
            raise ValueError("Packed operation batch must match attn2 branches.")
        if not isinstance(uses, Sequence) or not uses:
            raise ValueError("Packed operation masks require target uses.")
        composition = tuple(use.composition_index for use in uses)
        if composition != tuple(sorted(composition)):
            raise ValueError("Packed operation uses must follow composition order.")

    @staticmethod
    def _packed_use_mask(
        execution: UnetAttn2Execution,
        *,
        use: RegionalOperationMaskUse,
        inputs: torch.Tensor,
        spatial: bool,
    ) -> torch.Tensor:
        """Return one use mask in exact compact branch-segment order."""

        if use.region_index >= int(execution.query_masks.shape[0]):
            raise ValueError("Packed operation use references an unavailable region.")
        source_gate = REGIONAL_OPERATION_BRANCH_GATE_RESOLVER.resolve(
            execution.contexts,
            branch=use.branch,
            authority=inputs,
        )
        segments: list[torch.Tensor] = []
        for segment in execution.branches.segments:
            count = int(segment.source_indices.shape[0])
            if segment.key.region_index != use.region_index:
                segments.append(inputs.new_zeros((count, int(inputs.shape[1]), 1)))
                continue
            gate = source_gate.index_select(0, segment.source_indices).reshape(
                count,
                1,
                1,
            )
            if spatial:
                mask = execution.query_masks[use.region_index].index_select(
                    0,
                    segment.source_indices,
                )
                segments.append(mask.unsqueeze(-1) * gate)
            else:
                segments.append(gate.expand(-1, int(inputs.shape[1]), -1))
        return torch.cat(tuple(segments))


STANDARD_UNET_PACKED_OPERATION_MASK_RESOLVER = StandardUnetPackedOperationMaskResolver()
