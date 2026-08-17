# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered benchmark snapshots for mixed conditioning values."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from tools.attention_coupling_benchmark.comfy_probe.conditioning_batch_snapshot import (
    SnapshotConditioningBatchV3,
    snapshot_conditioning_value,
)


def test_snapshot_preserves_batch_and_conditioning_entry_order() -> None:
    """Retain exact global/region and scheduled-entry tensor identities."""

    global_tensor = torch.tensor([[[1.0]]])
    region_first = torch.tensor([[[2.0]]])
    region_second = torch.tensor([[[3.0]]])
    value = ConditioningBatch(
        (
            [[global_tensor, {"name": "global"}]],
            [
                [region_first, {"start_percent": 0.0}],
                [region_second, {"start_percent": 0.5}],
            ],
        )
    )

    snapshot = snapshot_conditioning_value(value)

    assert snapshot["kind"] == "conditioning_batch"
    batches = snapshot["batch_entries"]
    assert isinstance(batches, list)
    assert [batch["batch_index"] for batch in batches] == [0, 1]
    region_entries = batches[1]["conditioning_entries"]
    assert [entry["entry_index"] for entry in region_entries] == [0, 1]
    assert (
        region_entries[0]["tensor"]["float32_sha256"]
        != region_entries[1]["tensor"]["float32_sha256"]
    )


def test_snapshot_accepts_plain_conditioning_and_rejects_malformed_entries() -> None:
    """Distinguish broadcast conditioning and fail on invalid tensor metadata pairs."""

    snapshot = snapshot_conditioning_value([[torch.zeros((1, 2, 3)), {}]])
    assert snapshot["kind"] == "conditioning"
    batch_entries = snapshot["batch_entries"]
    assert isinstance(batch_entries, list)
    assert len(batch_entries) == 1

    with pytest.raises(TypeError, match="tensor or metadata"):
        snapshot_conditioning_value([[object(), {}]])
    with pytest.raises(TypeError, match="conditioning sequence"):
        snapshot_conditioning_value(object())


def test_snapshot_node_is_output_only_and_dev_only() -> None:
    """Keep the observation node outside the product export surface."""

    schema = SnapshotConditioningBatchV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.SnapshotConditioningBatch"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True
