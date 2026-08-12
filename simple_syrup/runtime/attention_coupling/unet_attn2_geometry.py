# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve installed UNet attn2 metadata to exact query geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_mask_bank import RegionalMaskBank
from ...domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from ..regional_attention_diagnostic_values import RegionalAttentionQueryGeometry
from ..spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


@dataclass(frozen=True, slots=True)
class StandardUnetAttn2Geometry:
    """Retain one validated UNet layer identity and spatial query contract."""

    query: RegionalAttentionQueryGeometry
    layout: SpatialBatchLayout
    original_height: int
    original_width: int
    block: tuple[str, int]
    block_index: int
    transformer_index: int


class StandardUnetAttn2GeometryResolver:
    """Parse only installed Comfy callback metadata and layout identity."""

    def resolve(
        self,
        query: torch.Tensor,
        contexts: BatchedRegionalAttentionContexts,
        mask_bank: RegionalMaskBank,
        extra_options: dict[str, Any],
    ) -> StandardUnetAttn2Geometry:
        """Return exact rectangular query and model-call spatial geometry."""

        if not isinstance(query, torch.Tensor) or query.ndim != 3:
            raise ValueError("Standard UNet attn2 query must use BxQxD layout.")
        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Standard UNet geometry requires aligned contexts.")
        if not isinstance(mask_bank, RegionalMaskBank):
            raise TypeError("Standard UNet geometry requires a mask bank.")
        if not isinstance(extra_options, dict):
            raise TypeError("Standard UNet attn2 extra_options must be a dictionary.")
        activations = self._shape(
            extra_options.get("activations_shape"),
            name="activations_shape",
        )
        original = self._shape(
            extra_options.get("original_shape"),
            name="original_shape",
        )
        batch, _, query_height, query_width = activations
        if batch != int(query.shape[0]) or batch != int(contexts.base_context.shape[0]):
            raise ValueError(
                "Standard UNet attn2 activation batch must match query and contexts."
            )
        if int(query.shape[1]) != query_height * query_width:
            raise ValueError(
                "Standard UNet attn2 token count must match activations H/W."
            )
        if original[0] != batch:
            raise ValueError(
                "Standard UNet original batch must match the active query batch."
            )
        block = self._block(extra_options.get("block"))
        block_index = self._nonnegative_index(
            extra_options.get("block_index"),
            name="block_index",
        )
        transformer_index = self._nonnegative_index(
            extra_options.get("transformer_index"),
            name="transformer_index",
        )
        published_layout = self._published_layout(extra_options)
        layout = published_layout or self._full_layout(
            mask_bank,
            contexts,
            original_height=original[2],
            original_width=original[3],
        )
        if published_layout is not None:
            self._validate_published_layout(
                published_layout,
                query_batch=batch,
                original_height=original[2],
                original_width=original[3],
            )
        return StandardUnetAttn2Geometry(
            query=RegionalAttentionQueryGeometry(
                batch,
                1,
                query_height,
                query_width,
                published_layout,
            ),
            layout=layout,
            original_height=original[2],
            original_width=original[3],
            block=block,
            block_index=block_index,
            transformer_index=transformer_index,
        )

    @staticmethod
    def _shape(value: object, *, name: str) -> tuple[int, int, int, int]:
        """Narrow one installed BCHW metadata sequence."""

        if not isinstance(value, list | tuple) or len(value) != 4:
            raise TypeError(f"Standard UNet {name} must be a BCHW sequence.")
        values: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(
                    f"Standard UNet {name} dimensions must be positive integers."
                )
            values.append(item)
        return values[0], values[1], values[2], values[3]

    @staticmethod
    def _block(value: object) -> tuple[str, int]:
        """Narrow installed input/middle/output block metadata."""

        if not isinstance(value, list | tuple) or len(value) != 2:
            raise TypeError("Standard UNet block must contain kind and index.")
        kind, index = value
        if kind not in {"input", "middle", "output"}:
            raise ValueError("Standard UNet block kind is unsupported.")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Standard UNet block index must be non-negative.")
        return kind, index

    @staticmethod
    def _nonnegative_index(value: object, *, name: str) -> int:
        """Narrow one installed non-negative integer index."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Standard UNet {name} must be an integer.")
        if value < 0:
            raise ValueError(f"Standard UNet {name} must be non-negative.")
        return value

    @staticmethod
    def _published_layout(
        extra_options: dict[str, Any],
    ) -> SpatialBatchLayout | None:
        """Return the exact optional SimpleSyrup spatial layout."""

        namespace = extra_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if namespace is None:
            return None
        if not isinstance(namespace, dict):
            raise TypeError("Standard UNet SimpleSyrup namespace must be a dictionary.")
        layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if layout is None:
            return None
        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError(
                "Standard UNet spatial layout must be a SpatialBatchLayout."
            )
        return layout

    @staticmethod
    def _full_layout(
        mask_bank: RegionalMaskBank,
        contexts: BatchedRegionalAttentionContexts,
        *,
        original_height: int,
        original_width: int,
    ) -> SpatialBatchLayout:
        """Represent an unmodified full-canvas UNet model call."""

        if (original_height, original_width) != (
            mask_bank.canvas_height,
            mask_bank.canvas_width,
        ):
            raise ValueError(
                "Full-context UNet original H/W must match the regional mask canvas."
            )
        return SpatialBatchLayout(
            mask_bank.canvas_width,
            mask_bank.canvas_height,
            (
                SpatialView(
                    SpatialViewKind.FULL,
                    0,
                    0,
                    mask_bank.canvas_width,
                    mask_bank.canvas_height,
                    mask_bank.canvas_width,
                    mask_bank.canvas_height,
                ),
            ),
            contexts.latent_batch_size,
        )

    @staticmethod
    def _validate_published_layout(
        layout: SpatialBatchLayout,
        *,
        query_batch: int,
        original_height: int,
        original_width: int,
    ) -> None:
        """Require published view batches to match the active UNet tensor."""

        if layout.expanded_batch_size != query_batch:
            raise ValueError(
                "Standard UNet spatial layout batch must match the active query."
            )
        mismatched = tuple(
            index
            for index, view in enumerate(layout.views)
            if (view.model_height, view.model_width)
            != (original_height, original_width)
        )
        if mismatched:
            raise ValueError(
                "Standard UNet original H/W must match every layout model view; "
                f"mismatched view indices {mismatched}."
            )


STANDARD_UNET_ATTN2_GEOMETRY_RESOLVER = StandardUnetAttn2GeometryResolver()
