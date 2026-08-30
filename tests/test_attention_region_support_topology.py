# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify adaptive semantic support selection from evidence topology."""

import torch

from simple_syrup.services.attention_region_support_topology import (
    ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY,
)


def test_peak_dominated_broad_evidence_prefers_concentrated_consensus() -> None:
    """Prefer a compact stable core over a much broader weak primary field."""

    primary = torch.zeros(10, 10)
    primary[3:7, :] = 0.3
    primary[4, 4:6] = 1.0
    concentrated = torch.zeros(10, 10)
    concentrated[4, 4:6] = 1.0

    assert ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=True,
    )


def test_compact_dominant_semantic_evidence_uses_relative_core() -> None:
    """Tighten a compact semantic field even when it stays below 15% coverage."""

    primary = torch.zeros(20, 20)
    primary[6:13, 6:13] = 0.3
    primary[6:12, 7:12] = 1.0
    concentrated = torch.zeros_like(primary)
    concentrated[6:12, 7:12] = 1.0

    assert ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=True,
    )


def test_thin_semantic_evidence_preserves_connected_extent() -> None:
    """Keep a narrow semantic extension around its stronger center."""

    primary = torch.zeros(20, 20)
    primary[:, 9:11] = 0.3
    primary[7:12, 9:11] = 1.0
    concentrated = torch.zeros_like(primary)
    concentrated[7:12, 9:11] = 1.0

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=True,
    )


def test_distributed_semantic_evidence_preserves_sparse_instances() -> None:
    """Keep dispersed semantic instances rather than collapsing to one peak."""

    primary = torch.zeros(20, 20)
    for row, column in ((1, 1), (1, 9), (2, 16), (8, 4), (10, 14), (16, 2)):
        primary[row : row + 2, column : column + 2] = 0.3
    primary[1:3, 1:3] = 1.0
    concentrated = torch.zeros_like(primary)
    concentrated[1:3, 1:3] = 1.0

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=True,
    )


def test_compact_dominant_geometry_uses_relative_core_below_frame_threshold() -> None:
    """Tighten a compact field whose weak expansion stays below 15% of the frame."""

    primary = torch.zeros(20, 20, dtype=torch.bool)
    primary[6:13, 6:13] = True
    concentrated = torch.zeros_like(primary)
    concentrated[8:11, 8:11] = True

    assert ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_geometry(
        primary,
        concentrated,
        geometry_recall=0.85,
    )


def test_thin_extended_geometry_preserves_weak_connected_extent() -> None:
    """Keep a narrow extension even when its area greatly exceeds its strong core."""

    primary = torch.zeros(20, 20, dtype=torch.bool)
    primary[:, 9:11] = True
    concentrated = torch.zeros_like(primary)
    concentrated[7:12, 9:11] = True

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_geometry(
        primary,
        concentrated,
        geometry_recall=0.85,
    )


def test_distributed_geometry_preserves_multiple_sparse_instances() -> None:
    """Keep dispersed components rather than collapsing them to one strong instance."""

    primary = torch.zeros(20, 20, dtype=torch.bool)
    for row, column in ((1, 1), (1, 9), (2, 16), (8, 4), (10, 14), (16, 2)):
        primary[row : row + 2, column : column + 2] = True
    concentrated = torch.zeros_like(primary)
    concentrated[1:3, 1:3] = True

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_geometry(
        primary,
        concentrated,
        geometry_recall=0.85,
    )


def test_explicit_full_geometry_recall_bypasses_relative_compactness() -> None:
    """Honor an explicit maximum-recall request for compact weak geometry."""

    primary = torch.zeros(20, 20, dtype=torch.bool)
    primary[6:13, 6:13] = True
    concentrated = torch.zeros_like(primary)
    concentrated[8:11, 8:11] = True

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_geometry(
        primary,
        concentrated,
        geometry_recall=1.0,
    )


def test_broad_stable_evidence_keeps_primary_consensus() -> None:
    """Keep genuinely broad evidence when its stronger consensus remains broad."""

    primary = torch.zeros(10, 10)
    primary[2:8, :] = 0.8
    concentrated = primary.clone()

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=True,
    )


def test_standard_model_evidence_does_not_use_anima_adaptation() -> None:
    """Leave non-adaptive model-family evidence on its established path."""

    primary = torch.ones(10, 10)
    concentrated = torch.zeros(10, 10)
    concentrated[4, 4] = 1.0

    assert not ATTENTION_REGION_SUPPORT_TOPOLOGY_POLICY.prefers_concentrated_evidence(
        primary,
        concentrated,
        0.15,
        adaptive=False,
    )
