# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile the exact pinned Anima regional-LoRA execution profiles."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.profiling import (
    capture_performance_profiles,
)
from tools.anima_regional_lora_performance.suite import performance_suite_session
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
)


def main() -> int:
    """Capture all profiles under one guaranteed model-lifecycle session."""

    arguments = _arguments()
    manifest = default_manifest()
    output_directory = arguments.output_root / _timestamp()
    with performance_suite_session(manifest) as suite:
        result_path = capture_performance_profiles(
            suite,
            output_directory=output_directory,
            call_count=arguments.calls,
        )
    print(f"result={result_path}")
    return 0


def _arguments() -> argparse.Namespace:
    """Parse explicit profile depth and external evidence-directory controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=_profile_calls, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_benchmark_artifact_root(
            "anima-regional-prompting-v1/p10.1/profiles"
        ),
    )
    return parser.parse_args()


def _profile_calls(value: str) -> int:
    """Parse a non-empty prefix of the fixed 30-call trajectory."""

    parsed = int(value)
    if not 1 <= parsed <= 30:
        raise argparse.ArgumentTypeError("calls must be between 1 and 30")
    return parsed


def _timestamp() -> str:
    """Return one Windows-safe UTC evidence directory name."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
