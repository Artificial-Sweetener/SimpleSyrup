# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify backend-neutral regional execution diagnostics."""

from __future__ import annotations

import ast
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime import regional_attention_diagnostics as diagnostics_module
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionQueryGeometry,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


def test_common_diagnostics_report_canonical_layout_cfg_and_work() -> None:
    """Build exact JSON-safe fields from model-neutral authorities."""

    mask_bank = _mask_bank()
    source_masks = mask_bank.conditioning_masks.clone()
    contexts = _contexts()
    layout = _tile_layout()
    builder = RegionalAttentionDiagnosticsBuilder(
        mask_bank,
        backend="test.standard_unet",
    )

    snapshot = builder.build(
        contexts,
        RegionalAttentionQueryGeometry(4, 1, 1, 2, layout),
        layout,
        transformer_options=_options(layout, [1, 0, 1, 0], with_call_values=True),
    )

    assert snapshot.strategy == "attention_coupling"
    assert snapshot.backend == "test.standard_unet"
    assert snapshot.region_count == 2
    assert snapshot.coverage_class == "mixed"
    assert snapshot.active_region_indices == (0, 1)
    assert snapshot.uncovered_fraction == 0.25
    assert snapshot.overlap_fraction == 0.5
    assert snapshot.spatial_mode == "tile"
    assert snapshot.query_token_count == 2
    assert snapshot.active_branches == ("negative", "positive")
    assert snapshot.positive_chunk_count == 2
    assert snapshot.negative_chunk_count == 2
    assert snapshot.cross_attention_branch_multiplier == 3.0
    assert snapshot.denoiser_call_multiplier == 1.0
    assert snapshot.sampling_sigma == 0.5
    assert len(snapshot.conditioning_uuids) == 4
    assert [entry.strengths for entry in snapshot.regional_entries] == [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    assert all(entry.active for entry in snapshot.regional_entries)
    assert [chunk.branch for chunk in snapshot.chunks] == [
        "negative",
        "positive",
        "negative",
        "positive",
    ]
    assert [view.source_x for view in snapshot.views] == [0, 2]
    assert snapshot.to_log_fields()["estimated_work"] == {
        "cross_attention_branch_multiplier": 3.0,
        "cross_attention_formula": "base_plus_region_count",
        "denoiser_call_multiplier": 1.0,
    }
    json.dumps(snapshot.to_log_fields())
    assert torch.equal(mask_bank.conditioning_masks, source_masks)


def test_common_diagnostics_classify_all_base_without_adapter_policy() -> None:
    """Retain zero-coverage regions while reporting one denoiser trajectory."""

    masks = torch.zeros(2, 1, 4)
    mask_bank = RegionalMaskBank(masks, masks.clone(), 4, 1)
    layout = _full_layout()

    snapshot = RegionalAttentionDiagnosticsBuilder(
        mask_bank,
        backend="test.standard_unet",
    ).build(
        _contexts(),
        RegionalAttentionQueryGeometry(4, 1, 1, 4, None),
        layout,
    )

    assert snapshot.coverage_class == "all_base"
    assert snapshot.active_region_indices == ()
    assert snapshot.cross_attention_branch_multiplier == 3.0
    assert snapshot.denoiser_call_multiplier == 1.0


def test_common_diagnostics_reject_layout_cfg_and_batch_permutations() -> None:
    """Fail closed when one model-call axis diverges from its authority."""

    builder = RegionalAttentionDiagnosticsBuilder(
        _mask_bank(),
        backend="test.standard_unet",
    )
    contexts = _contexts()
    layout = _tile_layout()
    geometry = RegionalAttentionQueryGeometry(4, 1, 1, 2, layout)

    with pytest.raises(ValueError, match="query geometry authority"):
        builder.build(contexts, geometry, _tile_layout())
    with pytest.raises(ValueError, match="published spatial layout"):
        builder.build(
            contexts,
            geometry,
            layout,
            transformer_options=_options(_tile_layout(), [1, 0, 1, 0]),
        )
    with pytest.raises(ValueError, match="cond_or_uncond order mismatch"):
        builder.build(
            contexts,
            geometry,
            layout,
            transformer_options=_options(layout, [0, 1, 0, 1]),
        )
    with pytest.raises(ValueError, match="batch-layout mismatch"):
        builder.build(
            contexts,
            RegionalAttentionQueryGeometry(3, 1, 1, 2, layout),
            layout,
        )
    bad_uuids = _options(layout, [1, 0, 1, 0])
    bad_uuids["uuids"] = [str(uuid.uuid4())]
    with pytest.raises(ValueError, match="UUID count"):
        builder.build(contexts, geometry, layout, transformer_options=bad_uuids)


def test_common_diagnostics_reject_malformed_model_call_values() -> None:
    """Fail closed on nonuniform sigma and non-v4 conditioning identities."""

    builder = RegionalAttentionDiagnosticsBuilder(
        _mask_bank(),
        backend="test.standard_unet",
    )
    contexts = _contexts()
    layout = _tile_layout()
    geometry = RegionalAttentionQueryGeometry(4, 1, 1, 2, layout)
    nonuniform = _options(layout, [1, 0, 1, 0])
    nonuniform["sigmas"] = torch.tensor([0.5, 0.25])
    with pytest.raises(ValueError, match="uniform"):
        builder.build(contexts, geometry, layout, transformer_options=nonuniform)
    invalid_uuid = _options(layout, [1, 0, 1, 0])
    invalid_uuid["uuids"] = [str(uuid.uuid1())] * 4
    with pytest.raises(ValueError, match="UUIDv4"):
        builder.build(contexts, geometry, layout, transformer_options=invalid_uuid)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: RegionalAttentionQueryGeometry(0, 1, 1, 1, None),
            "input_batch_size must be positive",
        ),
        (
            lambda: RegionalAttentionQueryGeometry(1, 0, 1, 1, None),
            "query_time must be positive",
        ),
        (
            lambda: RegionalAttentionQueryGeometry(1, 1, 0, 1, None),
            "query_height must be positive",
        ),
        (
            lambda: RegionalAttentionQueryGeometry(1, 1, 1, 0, None),
            "query_width must be positive",
        ),
        (
            lambda: RegionalAttentionQueryGeometry(True, 1, 1, 1, None),
            "input_batch_size must be an integer",
        ),
    ],
)
def test_common_query_geometry_rejects_invalid_axes(
    factory: Callable[[], object],
    message: str,
) -> None:
    """Reject malformed query geometry before snapshot construction."""

    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_common_diagnostics_source_has_no_anima_dependency() -> None:
    """Keep model-neutral diagnostics independent from Anima runtime owners."""

    source_path = Path(diagnostics_module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = tuple(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert all("anima" not in imported for imported in imports)


def _mask_bank() -> RegionalMaskBank:
    """Return overlapping, uncovered canonical mask authorities."""

    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.5, 0.0]],
            [[0.0, 0.5, 1.0, 0.0]],
        ]
    )
    return RegionalMaskBank(masks, masks.clone(), 4, 1)


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return reversed and repeated CFG chunks with two regional entries."""

    branches = (
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
    )
    return BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=tuple(
            RegionalAttentionChunkBatch(index, branch, index, index + 1)
            for index, branch in enumerate(branches)
        ),
        base_context=torch.tensor([[[-1.0]], [[1.0]], [[-1.0]], [[1.0]]]),
        regions=tuple(
            BatchedRegionalAttentionRegion(
                region_index,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.full((4, 1, 1), float(region_index + 2)),
                        (1.0, 1.0, 1.0, 1.0),
                    ),
                ),
            )
            for region_index in range(2)
        ),
    )


def _tile_layout() -> SpatialBatchLayout:
    """Return two ordered tile views over a four-token canvas."""

    return SpatialBatchLayout(
        4,
        1,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 1, 2, 1),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 1, 2, 1),
        ),
        2,
    )


def _full_layout() -> SpatialBatchLayout:
    """Return one synthetic full layout before CFG chunk expansion."""

    return SpatialBatchLayout(
        4,
        1,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 4, 1, 4, 1),),
        1,
    )


def _options(
    layout: SpatialBatchLayout,
    selectors: list[int],
    *,
    with_call_values: bool = False,
) -> dict[str, object]:
    """Return exact published layout and CFG metadata."""

    values: dict[str, object] = {
        "cond_or_uncond": selectors,
        SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {SPATIAL_BATCH_LAYOUT_KEY: layout},
    }
    if with_call_values:
        values["sigmas"] = torch.full((len(selectors),), 0.5)
        values["uuids"] = [str(uuid.uuid4()) for _ in selectors]
    return values
