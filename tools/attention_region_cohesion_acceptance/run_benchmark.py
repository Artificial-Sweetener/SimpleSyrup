# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark matched warmed generation with and without attention capture."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark_artifacts import (
    BenchmarkSummary,
    compose_benchmark_sheet,
    pixels_identical,
)
from .benchmark_workflow import build_benchmark_pair, build_benchmark_proof
from .client import ExecutionResult, execute_with_metrics
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Describe one fixed-source performance case."""

    label: str
    source: Path
    concept: str
    seed_base: int


def main() -> None:
    """Run matched warmed pairs per model and write proof artifacts."""

    summaries: list[BenchmarkSummary] = []
    manifest_cases: list[dict[str, object]] = []
    for case in _cases():
        summary, measurements = _benchmark(case)
        summaries.append(summary)
        manifest_cases.append(
            {
                "label": case.label,
                "concept": case.concept,
                "baseline_seconds": [item.elapsed_seconds for item in measurements[0]],
                "capture_seconds": [item.elapsed_seconds for item in measurements[1]],
                "summary": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(summary).items()
                },
            }
        )
    sheet = PROOF_ROOT / "attention_capture_performance.png"
    compose_benchmark_sheet(tuple(summaries), sheet)
    destination = PROOF_ROOT / "attention_capture_performance.json"
    destination.write_text(
        json.dumps(
            {"server": SERVER, "cases": manifest_cases, "sheet": str(sheet)}, indent=2
        ),
        encoding="utf-8",
    )
    print(destination)


def _benchmark(
    case: BenchmarkCase,
) -> tuple[BenchmarkSummary, tuple[list[ExecutionResult], list[ExecutionResult]]]:
    """Benchmark one model with alternating same-seed execution order."""

    baseline_results: list[ExecutionResult] = []
    capture_results: list[ExecutionResult] = []
    representative: tuple[ExecutionResult, ExecutionResult] | None = None
    for iteration in range(3):
        seed = case.seed_base + iteration
        prefix = (
            "simple_syrup_attention_cohesion_proof/benchmark/"
            f"{_slug(case.label)}/{iteration}"
        )
        baseline, capture = build_benchmark_pair(
            case.source,
            concept=case.concept,
            seed=seed,
            output_prefix=prefix,
        )
        ordered = (
            ("baseline", baseline, baseline_results),
            ("capture", capture, capture_results),
        )
        if iteration % 2:
            ordered = tuple(reversed(ordered))
        current: dict[str, ExecutionResult] = {}
        for label, graph, destination in ordered:
            result = execute_with_metrics(SERVER, graph, OUTPUT_ROOT)
            destination.append(result)
            current[label] = result
        if iteration == 2:
            representative = (current["baseline"], current["capture"])
    assert representative is not None
    baseline_result, capture_result = representative
    proof = build_benchmark_proof(
        case.source,
        concept=case.concept,
        seed=case.seed_base + 2,
        output_prefix=(
            "simple_syrup_attention_cohesion_proof/benchmark/"
            f"{_slug(case.label)}/representative"
        ),
    )
    proof_result = execute_with_metrics(SERVER, proof, OUTPUT_ROOT)
    return (
        BenchmarkSummary(
            label=case.label,
            baseline_seconds=statistics.median(
                result.elapsed_seconds for result in baseline_results
            ),
            capture_seconds=statistics.median(
                result.elapsed_seconds for result in capture_results
            ),
            pixels_identical=pixels_identical(
                baseline_result.outputs["900"],
                capture_result.outputs["900"],
            ),
            image_path=capture_result.outputs["900"],
            mask_path=proof_result.outputs["1131"],
        ),
        (baseline_results, capture_results),
    )


def _cases() -> tuple[BenchmarkCase, ...]:
    """Return fixed SDXL and Anima benchmark sources."""

    source_root = OUTPUT_ROOT / "simple_syrup_attention_acceptance_v2"
    return (
        BenchmarkCase(
            "SDXL / Illustrious Amanatsu",
            source_root / "sdxl/final_many_image_00002_.png",
            "hair",
            991_700,
        ),
        BenchmarkCase(
            "Hassaku Anima",
            source_root / "anima/final_many_image_00002_.png",
            "holding cat",
            992_700,
        ),
    )


def _slug(value: str) -> str:
    """Return a stable output-folder segment."""

    return "_".join(value.lower().replace("/", " ").split())


if __name__ == "__main__":
    main()
