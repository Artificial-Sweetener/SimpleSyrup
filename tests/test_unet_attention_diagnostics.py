"""Verify standard-UNet structured resolution diagnostic emission."""

from __future__ import annotations

import json
import logging

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.attention_coupling.unet_attention_diagnostics import (
    StandardUnetAttentionDiagnosticsEmitter,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_geometry import (
    StandardUnetAttn2Geometry,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionChunkDiagnostics,
    RegionalAttentionExecutionDiagnostics,
    RegionalAttentionQueryGeometry,
    RegionalAttentionRegionCoverageDiagnostics,
    RegionalAttentionViewDiagnostics,
)


class _RecordHandler(logging.Handler):
    """Retain emitted log records without formatter mutation."""

    def __init__(self) -> None:
        """Initialize one empty record list."""

        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one structured record."""

        self.records.append(record)


def test_unet_diagnostics_emit_exact_json_safe_resolution_fields() -> None:
    """Publish the shared snapshot beside exact installed layer geometry."""

    handler = _RecordHandler()
    logger = logging.Logger("tests.unet.emitter", level=logging.INFO)
    logger.addHandler(handler)

    StandardUnetAttentionDiagnosticsEmitter(logger).emit(_snapshot(), _geometry())

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.__dict__["operation"] == "unet_attention_coupling.resolve"
    assert record.__dict__["unet_layer"] == {
        "block_kind": "middle",
        "block_number": 0,
        "block_index": 1,
        "transformer_index": 7,
        "query_height": 2,
        "query_width": 3,
    }
    fields = record.__dict__["regional_diagnostics"]
    assert isinstance(fields, dict)
    assert fields["backend"] == "test.standard_unet"
    assert fields["estimated_work"] == {
        "cross_attention_branch_multiplier": 2.0,
        "cross_attention_formula": "base_plus_region_count",
        "denoiser_call_multiplier": 1.0,
    }
    json.dumps(fields)


def test_unet_diagnostics_skip_record_construction_when_info_is_disabled() -> None:
    """Avoid structured serialization work when INFO diagnostics are disabled."""

    handler = _RecordHandler()
    logger = logging.Logger("tests.unet.disabled", level=logging.WARNING)
    logger.addHandler(handler)

    StandardUnetAttentionDiagnosticsEmitter(logger).emit(_snapshot(), _geometry())

    assert handler.records == []


def _geometry() -> StandardUnetAttn2Geometry:
    """Return one exact rectangular middle-block geometry."""

    layout = _layout()
    return StandardUnetAttn2Geometry(
        RegionalAttentionQueryGeometry(1, 1, 2, 3, layout),
        layout,
        4,
        6,
        ("middle", 0),
        1,
        7,
    )


def _snapshot() -> RegionalAttentionExecutionDiagnostics:
    """Return one complete backend-neutral diagnostic snapshot."""

    return RegionalAttentionExecutionDiagnostics(
        strategy="attention_coupling",
        backend="test.standard_unet",
        region_count=1,
        coverage_class="mixed",
        active_region_indices=(0,),
        canonical_width=6,
        canonical_height=4,
        region_coverage=(RegionalAttentionRegionCoverageDiagnostics(0, 0.5, 0.5, 1.0),),
        uncovered_fraction=0.5,
        overlap_fraction=0.0,
        spatial_mode="full",
        views=(RegionalAttentionViewDiagnostics(0, "full", 0, 0, 6, 4, 6, 4),),
        query_time=1,
        query_height=2,
        query_width=3,
        query_token_count=6,
        input_batch_size=1,
        latent_batch_size=1,
        layout_input_batch_size=1,
        expanded_view_batch_size=1,
        active_branches=("positive",),
        positive_chunk_count=1,
        negative_chunk_count=0,
        chunks=(RegionalAttentionChunkDiagnostics(0, "positive", 0, 1),),
        sampling_sigma=1.0,
        conditioning_uuids=(),
        regional_entries=(),
        cross_attention_branch_multiplier=2.0,
        denoiser_call_multiplier=1.0,
    )


def _layout() -> SpatialBatchLayout:
    """Return one full-source six-by-four view."""

    return SpatialBatchLayout(
        6,
        4,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 6, 4, 6, 4),),
        1,
    )
