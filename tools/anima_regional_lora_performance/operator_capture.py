# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode complete PyTorch profiler aggregates into immutable evidence rows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast


class ProfilerAggregateEvent(Protocol):
    """Describe the stable aggregate fields consumed from PyTorch profiling."""

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


@dataclass(frozen=True, slots=True)
class PerformanceOperatorObservation:
    """Retain one complete aggregate operator row without filtering."""

    name: str
    count: int
    self_cpu_time_us: float
    total_cpu_time_us: float
    self_device_time_us: float
    total_device_time_us: float
    self_cpu_memory_bytes: int
    total_cpu_memory_bytes: int
    self_device_memory_bytes: int
    total_device_memory_bytes: int


def decode_operator_observations(
    events: object,
) -> tuple[PerformanceOperatorObservation, ...]:
    """Decode every aggregate event and sort it deterministically."""

    if not isinstance(events, Iterable):
        raise TypeError("PyTorch profiler key averages must be iterable.")
    values = tuple(cast(ProfilerAggregateEvent, event) for event in events)
    observations = tuple(
        PerformanceOperatorObservation(
            name=str(event.key),
            count=int(event.count),
            self_cpu_time_us=float(event.self_cpu_time_total),
            total_cpu_time_us=float(event.cpu_time_total),
            self_device_time_us=float(event.self_device_time_total),
            total_device_time_us=float(event.device_time_total),
            self_cpu_memory_bytes=int(event.self_cpu_memory_usage),
            total_cpu_memory_bytes=int(event.cpu_memory_usage),
            self_device_memory_bytes=int(event.self_device_memory_usage),
            total_device_memory_bytes=int(event.device_memory_usage),
        )
        for event in values
    )
    return tuple(
        sorted(
            observations,
            key=lambda value: (
                -value.self_device_time_us,
                -value.self_cpu_time_us,
                value.name,
            ),
        )
    )
