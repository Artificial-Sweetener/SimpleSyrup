"""Verify complete deterministic PyTorch aggregate decoding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.anima_regional_lora_performance.operator_capture import (
    decode_operator_observations,
)


@dataclass(frozen=True, slots=True)
class _Event:
    """Expose the complete aggregate event protocol."""

    key: str
    count: int
    self_cpu_time_total: float
    cpu_time_total: float
    self_device_time_total: float
    device_time_total: float
    self_cpu_memory_usage: int
    cpu_memory_usage: int
    self_device_memory_usage: int
    device_memory_usage: int


def test_operator_capture_retains_and_sorts_every_field() -> None:
    """Preserve raw rows while ordering by device, CPU, and name."""

    values = decode_operator_observations(
        (
            _Event("cpu", 2, 20.0, 30.0, 1.0, 2.0, 3, 4, 5, 6),
            _Event("cuda", 1, 10.0, 15.0, 40.0, 50.0, 7, 8, 9, 10),
        )
    )

    assert tuple(value.name for value in values) == ("cuda", "cpu")
    assert values[0].count == 1
    assert values[0].self_device_time_us == 40.0
    assert values[0].total_device_memory_bytes == 10
    assert values[1].self_cpu_memory_bytes == 3


def test_operator_capture_rejects_non_iterable_events() -> None:
    """Fail closed rather than silently publishing no profiler evidence."""

    with pytest.raises(TypeError, match="must be iterable"):
        decode_operator_observations(object())
