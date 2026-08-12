# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable JSON-safe Anima LoRA diagnostic values."""

from __future__ import annotations

from dataclasses import dataclass

from ..regional_attention_diagnostic_values import (
    RegionalAttentionChunkDiagnostics,
    RegionalAttentionEntryDiagnostics,
    RegionalAttentionExecutionDiagnostics,
    RegionalAttentionRegionCoverageDiagnostics,
    RegionalAttentionViewDiagnostics,
    regional_attention_log_fields,
)


@dataclass(frozen=True, slots=True)
class AnimaAdapterUseDiagnostics:
    """Describe ordered adapter ownership through a non-sensitive stable token."""

    composition_index: int
    adapter_token: str
    region_index: int
    branch: str
    target_count: int
    effective_strength: float
    active: bool
    pruning_reason: str | None

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe ordered adapter fields without its source identity."""

        return {
            "composition_index": self.composition_index,
            "adapter_token": self.adapter_token,
            "region_index": self.region_index,
            "branch": self.branch,
            "target_count": self.target_count,
            "effective_strength": self.effective_strength,
            "active": self.active,
            "pruning_reason": self.pruning_reason,
        }


@dataclass(frozen=True, slots=True)
class AnimaWorkEstimateDiagnostics:
    """Report explicit cardinality-based work estimates, never execution policy."""

    cross_attention_branch_multiplier: float
    low_rank_adapter_multiplier: float
    active_adapter_uses: int
    active_target_count: int
    target_use_count: int
    deduplicated_target_group_count: int
    compatible_projection_batch_count: int
    deduplicated_target_uses: int
    denoiser_call_multiplier: float

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe structured log fields and formula labels."""

        return {
            "cross_attention_branch_multiplier": (
                self.cross_attention_branch_multiplier
            ),
            "cross_attention_formula": "base_plus_region_count",
            "low_rank_adapter_multiplier": self.low_rank_adapter_multiplier,
            "low_rank_formula": "target_groups_divided_by_active_targets",
            "active_adapter_uses": self.active_adapter_uses,
            "active_target_count": self.active_target_count,
            "target_use_count": self.target_use_count,
            "deduplicated_target_group_count": self.deduplicated_target_group_count,
            "compatible_projection_batch_count": (
                self.compatible_projection_batch_count
            ),
            "deduplicated_target_uses": self.deduplicated_target_uses,
            "denoiser_call_multiplier": self.denoiser_call_multiplier,
        }


@dataclass(frozen=True, slots=True)
class AnimaRegionalExecutionDiagnostics:
    """Retain one common regional snapshot plus Anima LoRA execution values."""

    strategy: str
    backend: str
    region_count: int
    coverage_class: str
    active_region_indices: tuple[int, ...]
    canonical_width: int
    canonical_height: int
    region_coverage: tuple[RegionalAttentionRegionCoverageDiagnostics, ...]
    uncovered_fraction: float
    overlap_fraction: float
    spatial_mode: str
    views: tuple[RegionalAttentionViewDiagnostics, ...]
    query_time: int
    query_height: int
    query_width: int
    query_token_count: int
    input_batch_size: int
    latent_batch_size: int
    layout_input_batch_size: int
    expanded_view_batch_size: int
    active_branches: tuple[str, ...]
    positive_chunk_count: int
    negative_chunk_count: int
    chunks: tuple[RegionalAttentionChunkDiagnostics, ...]
    sampling_sigma: float | None
    conditioning_uuids: tuple[str, ...]
    regional_entries: tuple[RegionalAttentionEntryDiagnostics, ...]
    adapter_uses: tuple[AnimaAdapterUseDiagnostics, ...]
    prepared_cache_entries: int
    work: AnimaWorkEstimateDiagnostics

    @classmethod
    def from_regional(
        cls,
        regional: RegionalAttentionExecutionDiagnostics,
        *,
        adapter_uses: tuple[AnimaAdapterUseDiagnostics, ...],
        prepared_cache_entries: int,
        work: AnimaWorkEstimateDiagnostics,
    ) -> AnimaRegionalExecutionDiagnostics:
        """Combine one shared snapshot with Anima-only adapter diagnostics."""

        if not isinstance(regional, RegionalAttentionExecutionDiagnostics):
            raise TypeError("Anima diagnostics require a shared regional snapshot.")
        return cls(
            strategy=regional.strategy,
            backend=regional.backend,
            region_count=regional.region_count,
            coverage_class=regional.coverage_class,
            active_region_indices=regional.active_region_indices,
            canonical_width=regional.canonical_width,
            canonical_height=regional.canonical_height,
            region_coverage=regional.region_coverage,
            uncovered_fraction=regional.uncovered_fraction,
            overlap_fraction=regional.overlap_fraction,
            spatial_mode=regional.spatial_mode,
            views=regional.views,
            query_time=regional.query_time,
            query_height=regional.query_height,
            query_width=regional.query_width,
            query_token_count=regional.query_token_count,
            input_batch_size=regional.input_batch_size,
            latent_batch_size=regional.latent_batch_size,
            layout_input_batch_size=regional.layout_input_batch_size,
            expanded_view_batch_size=regional.expanded_view_batch_size,
            active_branches=regional.active_branches,
            positive_chunk_count=regional.positive_chunk_count,
            negative_chunk_count=regional.negative_chunk_count,
            chunks=regional.chunks,
            sampling_sigma=regional.sampling_sigma,
            conditioning_uuids=regional.conditioning_uuids,
            regional_entries=regional.regional_entries,
            adapter_uses=adapter_uses,
            prepared_cache_entries=prepared_cache_entries,
            work=work,
        )

    def to_log_fields(self) -> dict[str, object]:
        """Return the unchanged Anima structured diagnostic mapping."""

        fields = regional_attention_log_fields(self)
        fields["adapter_uses"] = [use.to_log_fields() for use in self.adapter_uses]
        fields["prepared_cache_entries"] = self.prepared_cache_entries
        fields["estimated_work"] = self.work.to_log_fields()
        return fields
