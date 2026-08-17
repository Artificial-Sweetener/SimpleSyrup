# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture selected P10.1 scaling operator traces and summaries."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profiling import (
    capture_scaling_profiles,
)
from tools.anima_regional_lora_performance.matrix_suite import scaling_suite_session
from tools.comfy_integration.default_paths import default_benchmark_artifact_root


def main() -> int:
    """Prepare one suite and capture every trace-declared matrix position."""

    arguments = _arguments()
    manifest = default_scaling_manifest()
    output_directory = arguments.output_root / _timestamp()
    with scaling_suite_session(manifest) as suite:
        result_path = capture_scaling_profiles(
            suite,
            output_directory=output_directory,
            call_count=arguments.calls,
        )
    print(f"result={result_path}")
    return 0


def _arguments() -> argparse.Namespace:
    """Parse bounded call count and external output-root controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=_positive_calls, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_benchmark_artifact_root(
            "anima-regional-prompting-v1/p10.1/scaling-profiles"
        ),
    )
    return parser.parse_args()


def _positive_calls(value: str) -> int:
    """Parse a positive profiling call count."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("calls must be positive")
    return parsed


def _timestamp() -> str:
    """Return one Windows-safe UTC evidence directory name."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
