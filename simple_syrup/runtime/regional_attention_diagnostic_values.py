# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable JSON-safe backend-neutral regional diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..domain.spatial_views import SpatialBatchLayout


@dataclass(frozen=True, slots=True)
class RegionalAttentionRegionCoverageDiagnostics:
    """Summarize one canonical conditioning mask without retaining pixels."""

    region_index: int
    nonzero_fraction: float
    mean_weight: float
    maximum_weight: float

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe structured log fields."""

        return {
            "region_index": self.region_index,
            "nonzero_fraction": self.nonzero_fraction,
            "mean_weight": self.mean_weight,
            "maximum_weight": self.maximum_weight,
        }


@dataclass(frozen=True, slots=True)
class RegionalAttentionViewDiagnostics:
    """Describe one exact active spatial view without tensor state."""

    view_index: int
    kind: str
    source_x: int
    source_y: int
    source_width: int
    source_height: int
    model_width: int
    model_height: int

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe structured log fields."""

        return {
            "view_index": self.view_index,
            "kind": self.kind,
            "source_x": self.source_x,
            "source_y": self.source_y,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "model_width": self.model_width,
            "model_height": self.model_height,
        }


@dataclass(frozen=True, slots=True)
class RegionalAttentionChunkDiagnostics:
    """Describe one exact aligned conditioning chunk."""

    chunk_index: int
    branch: str
    batch_start: int
    batch_stop: int

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe structured log fields."""

        return {
            "chunk_index": self.chunk_index,
            "branch": self.branch,
            "batch_start": self.batch_start,
            "batch_stop": self.batch_stop,
        }


@dataclass(frozen=True, slots=True)
class RegionalAttentionEntryDiagnostics:
    """Describe one selected regional entry and its aligned sample strengths."""

    region_index: int
    entry_index: int
    strengths: tuple[float, ...]
    active: bool

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe selected-entry fields."""

        return {
            "region_index": self.region_index,
            "entry_index": self.entry_index,
            "strengths": list(self.strengths),
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class RegionalAttentionQueryGeometry:
    """Describe one model-family query grid at the shared diagnostic boundary."""

    input_batch_size: int
    query_time: int
    query_height: int
    query_width: int
    spatial_layout: SpatialBatchLayout | None

    def __post_init__(self) -> None:
        """Require positive integer geometry and one optional typed layout."""

        for name, value in (
            ("input_batch_size", self.input_batch_size),
            ("query_time", self.query_time),
            ("query_height", self.query_height),
            ("query_width", self.query_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Regional diagnostic {name} must be an integer.")
            if value < 1:
                raise ValueError(f"Regional diagnostic {name} must be positive.")
        if self.spatial_layout is not None and not isinstance(
            self.spatial_layout,
            SpatialBatchLayout,
        ):
            raise TypeError(
                "Regional diagnostic spatial_layout must be a SpatialBatchLayout."
            )

    @property
    def query_token_count(self) -> int:
        """Return the flattened query length."""

        return self.query_time * self.query_height * self.query_width


class RegionalAttentionDiagnosticSnapshot(Protocol):
    """Expose fields shared by model-family regional diagnostic snapshots."""

    @property
    def strategy(self) -> str:
        """Return the selected regional strategy."""

        ...

    @property
    def backend(self) -> str:
        """Return the exact model-family backend identity."""

        ...

    @property
    def region_count(self) -> int:
        """Return the canonical region count."""

        ...

    @property
    def coverage_class(self) -> str:
        """Return the canonical activation classification."""

        ...

    @property
    def active_region_indices(self) -> tuple[int, ...]:
        """Return active canonical region indices."""

        ...

    @property
    def canonical_width(self) -> int:
        """Return canonical mask width."""

        ...

    @property
    def canonical_height(self) -> int:
        """Return canonical mask height."""

        ...

    @property
    def region_coverage(
        self,
    ) -> tuple[RegionalAttentionRegionCoverageDiagnostics, ...]:
        """Return scalar summaries for each canonical region."""

        ...

    @property
    def uncovered_fraction(self) -> float:
        """Return the uncovered canonical fraction."""

        ...

    @property
    def overlap_fraction(self) -> float:
        """Return the above-unit-overlap canonical fraction."""

        ...

    @property
    def spatial_mode(self) -> str:
        """Return the exact spatial view kind."""

        ...

    @property
    def views(self) -> tuple[RegionalAttentionViewDiagnostics, ...]:
        """Return ordered scalar spatial-view values."""

        ...

    @property
    def query_time(self) -> int:
        """Return query-grid time."""

        ...

    @property
    def query_height(self) -> int:
        """Return query-grid height."""

        ...

    @property
    def query_width(self) -> int:
        """Return query-grid width."""

        ...

    @property
    def query_token_count(self) -> int:
        """Return flattened query-token count."""

        ...

    @property
    def input_batch_size(self) -> int:
        """Return active model-input batch size."""

        ...

    @property
    def latent_batch_size(self) -> int:
        """Return source latent batch size."""

        ...

    @property
    def layout_input_batch_size(self) -> int:
        """Return layout source-batch size."""

        ...

    @property
    def expanded_view_batch_size(self) -> int:
        """Return view-expanded layout batch size."""

        ...

    @property
    def active_branches(self) -> tuple[str, ...]:
        """Return active CFG branches in first-observed order."""

        ...

    @property
    def positive_chunk_count(self) -> int:
        """Return positive CFG chunk count."""

        ...

    @property
    def negative_chunk_count(self) -> int:
        """Return negative CFG chunk count."""

        ...

    @property
    def chunks(self) -> tuple[RegionalAttentionChunkDiagnostics, ...]:
        """Return ordered scalar CFG chunk values."""

        ...

    @property
    def sampling_sigma(self) -> float | None:
        """Return the exact supplied model-call sigma when available."""

        ...

    @property
    def conditioning_uuids(self) -> tuple[str, ...]:
        """Return ordered Comfy conditioning UUIDs when available."""

        ...

    @property
    def regional_entries(self) -> tuple[RegionalAttentionEntryDiagnostics, ...]:
        """Return exact selected regional entry strengths."""

        ...


@dataclass(frozen=True, slots=True)
class RegionalAttentionExecutionDiagnostics:
    """Retain one complete backend-neutral regional execution snapshot."""

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
    cross_attention_branch_multiplier: float
    denoiser_call_multiplier: float

    def to_log_fields(self) -> dict[str, object]:
        """Return the complete common structured diagnostic mapping."""

        fields = regional_attention_log_fields(self)
        fields["estimated_work"] = {
            "cross_attention_branch_multiplier": (
                self.cross_attention_branch_multiplier
            ),
            "cross_attention_formula": "base_plus_region_count",
            "denoiser_call_multiplier": self.denoiser_call_multiplier,
        }
        return fields


def regional_attention_log_fields(
    snapshot: RegionalAttentionDiagnosticSnapshot,
) -> dict[str, object]:
    """Serialize shared fields for every model-family diagnostic snapshot."""

    return {
        "strategy": snapshot.strategy,
        "backend": snapshot.backend,
        "region_count": snapshot.region_count,
        "coverage_class": snapshot.coverage_class,
        "active_region_indices": list(snapshot.active_region_indices),
        "canonical_mask": {
            "width": snapshot.canonical_width,
            "height": snapshot.canonical_height,
            "regions": [item.to_log_fields() for item in snapshot.region_coverage],
            "uncovered_fraction": snapshot.uncovered_fraction,
            "overlap_fraction": snapshot.overlap_fraction,
        },
        "spatial_mode": snapshot.spatial_mode,
        "views": [view.to_log_fields() for view in snapshot.views],
        "query_grid": {
            "time": snapshot.query_time,
            "height": snapshot.query_height,
            "width": snapshot.query_width,
            "token_count": snapshot.query_token_count,
        },
        "batch_layout": {
            "input_batch_size": snapshot.input_batch_size,
            "latent_batch_size": snapshot.latent_batch_size,
            "layout_input_batch_size": snapshot.layout_input_batch_size,
            "expanded_view_batch_size": snapshot.expanded_view_batch_size,
            "active_branches": list(snapshot.active_branches),
            "positive_chunk_count": snapshot.positive_chunk_count,
            "negative_chunk_count": snapshot.negative_chunk_count,
            "chunks": [chunk.to_log_fields() for chunk in snapshot.chunks],
        },
        "model_call": {
            "sampling_sigma": snapshot.sampling_sigma,
            "conditioning_uuids": list(snapshot.conditioning_uuids),
        },
        "regional_conditioning_entries": [
            entry.to_log_fields() for entry in snapshot.regional_entries
        ],
    }
