# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture deterministic Python function costs for one benchmark call."""

from __future__ import annotations

import cProfile
import pstats
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypedDict, TypeVar, cast

Result = TypeVar("Result")


class PythonFunctionProfileRow(TypedDict):
    """Describe one serialized cProfile function aggregate."""

    file: str
    line: int
    function: str
    primitive_calls: int
    total_calls: int
    self_time_seconds: float
    cumulative_time_seconds: float


@dataclass(frozen=True, slots=True)
class PythonCallProfile(Generic[Result]):
    """Retain one call result and its cumulative Python function rows."""

    result: Result
    functions: tuple[PythonFunctionProfileRow, ...]


class PythonCallProfiler:
    """Own cProfile execution and stable row serialization."""

    def capture(self, call: Callable[[], Result]) -> PythonCallProfile[Result]:
        """Execute one callable and return its top cumulative function costs."""

        if not callable(call):
            raise TypeError("Python call profiler requires a callable.")
        profiler = cProfile.Profile()
        try:
            result = profiler.runcall(call)
        finally:
            profiler.disable()
        stats = pstats.Stats(profiler)
        raw_stats = cast(
            dict[tuple[str, int, str], tuple[int, int, float, float, object]],
            vars(stats)["stats"],
        )
        unsorted_rows: list[PythonFunctionProfileRow] = []
        for (
            filename,
            line,
            function,
        ), (
            primitive_calls,
            total_calls,
            self_time,
            cumulative_time,
            _callers,
        ) in raw_stats.items():
            unsorted_rows.append(
                {
                    "file": Path(filename).name,
                    "line": line,
                    "function": function,
                    "primitive_calls": primitive_calls,
                    "total_calls": total_calls,
                    "self_time_seconds": self_time,
                    "cumulative_time_seconds": cumulative_time,
                }
            )
        rows = tuple(
            sorted(
                unsorted_rows,
                key=lambda row: (
                    -row["cumulative_time_seconds"],
                    -row["self_time_seconds"],
                    row["file"],
                    row["line"],
                    row["function"],
                ),
            )[:250]
        )
        return PythonCallProfile(result, rows)


PYTHON_CALL_PROFILER = PythonCallProfiler()
