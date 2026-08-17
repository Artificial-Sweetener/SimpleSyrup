# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the complete focused P10.1 Anima regional LoRA scaling matrix."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_performance.artifacts import load_performance_artifacts
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_results import (
    evaluate_scaling_result,
    write_scaling_result,
)
from tools.anima_regional_lora_performance.matrix_runner import (
    ANIMA_REGIONAL_LORA_SCALING_RUNNER,
)
from tools.comfy_integration.default_paths import default_benchmark_artifact_root


def main() -> int:
    """Execute every matrix position and return failure on any missed gate."""

    arguments = _arguments()
    manifest = default_scaling_manifest(
        repeats=arguments.repeats,
        artifacts=load_performance_artifacts(arguments.artifact_inventory),
    )
    observations, environment = ANIMA_REGIONAL_LORA_SCALING_RUNNER.run(manifest)
    result = evaluate_scaling_result(manifest, observations)
    result_path = arguments.output_root / _timestamp() / "result.json"
    write_scaling_result(
        result_path,
        manifest=manifest,
        result=result,
        environment=environment,
    )
    for profile in result.profiles:
        overhead = (
            "n/a"
            if profile.overhead_percent is None
            else f"{profile.overhead_percent:.2f}%"
        )
        print(
            f"{profile.profile_id}: median={profile.median_runtime_ms:.2f} ms, "
            f"overhead={overhead}, cache={profile.cache_entries}, "
            f"passed={profile.passed}"
        )
    print(f"result={result_path}")
    return 0 if result.passed else 1


def _arguments() -> argparse.Namespace:
    """Parse exact repeat and external artifact-root controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--repeats", type=_at_least_three, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_benchmark_artifact_root(
            "anima-regional-prompting-v1/p10.1/scaling-matrix"
        ),
    )
    return parser.parse_args()


def _at_least_three(value: str) -> int:
    """Parse the minimum repeat count required for median evidence."""

    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("repeats must be at least 3")
    return parsed


def _timestamp() -> str:
    """Return one Windows-safe UTC evidence directory name."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
