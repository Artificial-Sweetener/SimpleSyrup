# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build backend-neutral diagnostics from canonical regional authorities."""

from __future__ import annotations

import torch

from ..domain.regional_attention import RegionalAttentionBranch
from ..domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ..domain.regional_mask_bank import RegionalMaskBank
from ..domain.spatial_views import SpatialBatchLayout, SpatialView
from ..masking.regional_mask_activation import RegionalMaskActivationClassifier
from .regional_attention_diagnostic_values import (
    RegionalAttentionChunkDiagnostics,
    RegionalAttentionEntryDiagnostics,
    RegionalAttentionExecutionDiagnostics,
    RegionalAttentionQueryGeometry,
    RegionalAttentionRegionCoverageDiagnostics,
    RegionalAttentionViewDiagnostics,
)
from .regional_attention_model_call_values import (
    RegionalAttentionModelCallValues,
    regional_attention_model_call_values,
)
from .spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


class RegionalAttentionDiagnosticsBuilder:
    """Derive shared snapshots without model-family or adapter policy."""

    def __init__(
        self,
        mask_bank: RegionalMaskBank,
        *,
        backend: str,
        activation_classifier: RegionalMaskActivationClassifier | None = None,
    ) -> None:
        """Precompute static canonical coverage once per prepared model."""

        if not isinstance(mask_bank, RegionalMaskBank):
            raise TypeError("Regional diagnostics require a canonical mask bank.")
        if not isinstance(backend, str) or not backend:
            raise ValueError("Regional diagnostics require a backend identity.")
        classifier = activation_classifier or RegionalMaskActivationClassifier()
        masks = mask_bank.conditioning_masks
        self._mask_bank = mask_bank
        self._backend = backend
        self._activation = classifier.classify(masks)
        self._coverage = self._coverage_diagnostics(masks)
        summed = masks.sum(dim=0)
        self._uncovered_fraction = self._fraction(summed == 0)
        self._overlap_fraction = self._fraction(summed > 1)

    @property
    def mask_bank(self) -> RegionalMaskBank:
        """Return the exact canonical mask authority used by this builder."""

        return self._mask_bank

    @property
    def backend(self) -> str:
        """Return the stable runtime backend identity."""

        return self._backend

    def build(
        self,
        contexts: BatchedRegionalAttentionContexts,
        query_geometry: RegionalAttentionQueryGeometry,
        layout: SpatialBatchLayout,
        *,
        transformer_options: object | None = None,
    ) -> RegionalAttentionExecutionDiagnostics:
        """Build one common call snapshot after exact metadata validation."""

        self.validate_call(
            contexts,
            query_geometry,
            layout,
            transformer_options=transformer_options,
        )
        chunks = tuple(
            RegionalAttentionChunkDiagnostics(
                chunk.chunk_index,
                chunk.branch.value,
                chunk.batch_start,
                chunk.batch_stop,
            )
            for chunk in contexts.chunks
        )
        branches = tuple(dict.fromkeys(chunk.branch for chunk in chunks))
        call_values = regional_attention_model_call_values(transformer_options)
        self._validate_call_values(call_values, contexts)
        return RegionalAttentionExecutionDiagnostics(
            strategy="attention_coupling",
            backend=self._backend,
            region_count=self._mask_bank.region_count,
            coverage_class=self._activation.coverage_class.value,
            active_region_indices=self._activation.active_region_indices,
            canonical_width=self._mask_bank.canvas_width,
            canonical_height=self._mask_bank.canvas_height,
            region_coverage=self._coverage,
            uncovered_fraction=self._uncovered_fraction,
            overlap_fraction=self._overlap_fraction,
            spatial_mode=layout.views[0].kind.value,
            views=self._view_diagnostics(layout),
            query_time=query_geometry.query_time,
            query_height=query_geometry.query_height,
            query_width=query_geometry.query_width,
            query_token_count=query_geometry.query_token_count,
            input_batch_size=query_geometry.input_batch_size,
            latent_batch_size=contexts.latent_batch_size,
            layout_input_batch_size=layout.input_batch_size,
            expanded_view_batch_size=layout.expanded_batch_size,
            active_branches=branches,
            positive_chunk_count=sum(
                chunk.branch == RegionalAttentionBranch.POSITIVE.value
                for chunk in chunks
            ),
            negative_chunk_count=sum(
                chunk.branch == RegionalAttentionBranch.NEGATIVE.value
                for chunk in chunks
            ),
            chunks=chunks,
            sampling_sigma=call_values.sampling_sigma,
            conditioning_uuids=call_values.conditioning_uuids,
            regional_entries=self._entry_diagnostics(contexts),
            cross_attention_branch_multiplier=float(
                1 + sum(len(region.entries) for region in contexts.regions)
            ),
            denoiser_call_multiplier=1.0,
        )

    def validate_call(
        self,
        contexts: BatchedRegionalAttentionContexts,
        query_geometry: RegionalAttentionQueryGeometry,
        layout: SpatialBatchLayout,
        *,
        transformer_options: object | None = None,
    ) -> None:
        """Reject layout, batch, and CFG metadata that diverge from authorities."""

        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Regional diagnostics require aligned contexts.")
        if not isinstance(query_geometry, RegionalAttentionQueryGeometry):
            raise TypeError("Regional diagnostics require typed query geometry.")
        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError("Regional diagnostics require a spatial batch layout.")
        if len(contexts.regions) != self._mask_bank.region_count:
            raise ValueError(
                "Regional diagnostics context count must match the mask bank."
            )
        if (layout.canvas_width, layout.canvas_height) != (
            self._mask_bank.canvas_width,
            self._mask_bank.canvas_height,
        ):
            raise ValueError(
                "Regional diagnostics spatial layout must match the mask canvas."
            )
        if query_geometry.spatial_layout is None:
            expected_batch = layout.expanded_batch_size * len(contexts.chunks)
        else:
            if layout is not query_geometry.spatial_layout:
                raise ValueError(
                    "Regional diagnostics layout must be the query geometry authority."
                )
            if layout.expanded_batch_size != int(contexts.base_context.shape[0]):
                raise ValueError(
                    "Regional diagnostics mask-layout mismatch: spatial layout "
                    f"expanded batch {layout.expanded_batch_size} must match aligned "
                    f"context batch {int(contexts.base_context.shape[0])}."
                )
            expected_batch = layout.expanded_batch_size
        if query_geometry.input_batch_size != expected_batch:
            raise ValueError(
                "Regional diagnostics batch-layout mismatch: expected view-major "
                f"batch {expected_batch}, observed "
                f"{query_geometry.input_batch_size}."
            )
        if transformer_options is None:
            return
        if not isinstance(transformer_options, dict):
            raise TypeError(
                "Regional diagnostics transformer_options must be a dictionary."
            )
        self._validate_published_layout(transformer_options, query_geometry)
        observed = transformer_options.get("cond_or_uncond")
        if observed is None:
            return
        if not isinstance(observed, list | tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in observed
        ):
            raise TypeError(
                "Regional diagnostics cond_or_uncond must be integer selectors."
            )
        expected = tuple(
            0 if chunk.branch is RegionalAttentionBranch.POSITIVE else 1
            for chunk in contexts.chunks
        )
        if tuple(observed) != expected:
            raise ValueError(
                "Regional diagnostics cond_or_uncond order mismatch: expected "
                f"view-major selectors {expected}, observed {tuple(observed)}."
            )
        self._validate_call_values(
            regional_attention_model_call_values(transformer_options),
            contexts,
        )

    @staticmethod
    def _validate_call_values(
        values: RegionalAttentionModelCallValues,
        contexts: BatchedRegionalAttentionContexts,
    ) -> None:
        """Require supplied UUIDs to align with exact selected CFG chunks."""

        if values.conditioning_uuids and len(values.conditioning_uuids) != len(
            contexts.chunks
        ):
            raise ValueError(
                "Regional diagnostic UUID count must match selected CFG chunks."
            )

    @classmethod
    def _entry_diagnostics(
        cls,
        contexts: BatchedRegionalAttentionContexts,
    ) -> tuple[RegionalAttentionEntryDiagnostics, ...]:
        """Flatten already-selected regional entries without schedule policy."""

        return tuple(
            RegionalAttentionEntryDiagnostics(
                region.region_index,
                entry.entry_index,
                tuple(cls._rounded(float(value)) for value in entry.strengths),
                any(value != 0.0 for value in entry.strengths),
            )
            for region in contexts.regions
            for entry in region.entries
        )

    @staticmethod
    def _validate_published_layout(
        transformer_options: dict[object, object],
        query_geometry: RegionalAttentionQueryGeometry,
    ) -> None:
        """Require published spatial metadata to retain exact identity."""

        namespace = transformer_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if namespace is None:
            return
        if not isinstance(namespace, dict):
            raise TypeError(
                "Regional diagnostics SimpleSyrup namespace must be a dictionary."
            )
        published = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if published is not None and published is not query_geometry.spatial_layout:
            raise ValueError(
                "Regional diagnostics published spatial layout does not match the "
                "authoritative query layout."
            )

    @classmethod
    def _coverage_diagnostics(
        cls,
        masks: torch.Tensor,
    ) -> tuple[RegionalAttentionRegionCoverageDiagnostics, ...]:
        """Summarize each canonical mask from the validated mask authority."""

        return tuple(
            RegionalAttentionRegionCoverageDiagnostics(
                region_index=index,
                nonzero_fraction=cls._fraction(mask > 0),
                mean_weight=cls._rounded(float(mask.mean().item())),
                maximum_weight=cls._rounded(float(mask.max().item())),
            )
            for index, mask in enumerate(masks)
        )

    @staticmethod
    def _view_diagnostics(
        layout: SpatialBatchLayout,
    ) -> tuple[RegionalAttentionViewDiagnostics, ...]:
        """Convert authoritative views into immutable scalar diagnostics."""

        return tuple(
            RegionalAttentionDiagnosticsBuilder._view(index, view)
            for index, view in enumerate(layout.views)
        )

    @staticmethod
    def _view(index: int, view: SpatialView) -> RegionalAttentionViewDiagnostics:
        """Convert one authoritative view into diagnostics."""

        return RegionalAttentionViewDiagnostics(
            index,
            view.kind.value,
            view.source_x,
            view.source_y,
            view.source_width,
            view.source_height,
            view.model_width,
            view.model_height,
        )

    @classmethod
    def _fraction(cls, values: torch.Tensor) -> float:
        """Return a stable six-decimal fraction for one boolean grid."""

        return cls._rounded(float(values.float().mean().item()))

    @staticmethod
    def _rounded(value: float) -> float:
        """Stabilize diagnostic floats without changing execution tensors."""

        return round(value, 6)
