# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the pinned full-context Anima regional LoRA performance gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.results import evaluate, write_result
from tools.anima_regional_lora_performance.runner import (
    ANIMA_REGIONAL_LORA_PERFORMANCE_RUNNER,
)


def main() -> int:
    """Execute the complete benchmark and return failure when a gate misses."""

    arguments = _arguments()
    manifest = default_manifest(repeats=arguments.repeats)
    observations, environment = ANIMA_REGIONAL_LORA_PERFORMANCE_RUNNER.run(manifest)
    result = evaluate(manifest, observations)
    output_directory = arguments.output_root / _timestamp()
    result_path = output_directory / "result.json"
    write_result(
        result_path,
        manifest=manifest,
        result=result,
        environment=environment,
    )
    for profile in result.profiles:
        print(
            f"{profile.profile_id}: median={profile.median_runtime_ms:.2f} ms, "
            f"overhead={profile.overhead_percent:.2f}%, "
            f"limit={profile.maximum_overhead_percent:.2f}%, "
            f"passed={profile.passed}"
        )
    print(f"result={result_path}")
    return 0 if result.passed else 1


def _arguments() -> argparse.Namespace:
    """Parse explicit repeat and external evidence-directory controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=_at_least_three, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p5.7"
        ),
    )
    return parser.parse_args()


def _at_least_three(value: str) -> int:
    """Parse the minimum repeat count required for a median gate."""

    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("repeats must be at least 3")
    return parsed


def _timestamp() -> str:
    """Return one Windows-safe UTC evidence directory name."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
