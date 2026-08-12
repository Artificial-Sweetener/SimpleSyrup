"""Verify deterministic benchmark tensor identity without source mutation."""

from __future__ import annotations

import pytest
import torch

from tools.attention_coupling_benchmark.comfy_probe.tensor_snapshot import (
    snapshot_tensor,
)


def test_snapshot_tensor_is_deterministic_and_preserves_source() -> None:
    """Record canonical content identity while retaining source dtype and values."""

    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    before = tensor.clone()

    first = snapshot_tensor(tensor)
    second = snapshot_tensor(tensor)

    assert first == second
    assert first["shape"] == [1, 2]
    assert first["dtype"] == "torch.bfloat16"
    assert isinstance(first["float32_sha256"], str)
    assert len(first["float32_sha256"]) == 64
    assert torch.equal(tensor, before)


def test_snapshot_tensor_rejects_non_tensor_values() -> None:
    """Fail closed rather than hashing an unrelated dynamic value."""

    with pytest.raises(TypeError, match="torch.Tensor"):
        snapshot_tensor(object())  # type: ignore[arg-type]
