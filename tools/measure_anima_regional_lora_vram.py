# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure isolated regional-LoRA profile VRAM through production execution."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_performance.isolated_measurement import (
    evaluate_isolated_vram_result,
    write_isolated_vram_result,
)
from tools.anima_regional_lora_performance.isolated_runner import (
    ANIMA_REGIONAL_LORA_ISOLATED_VRAM_RUNNER,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)


def main() -> int:
    """Run the complete isolated matrix and publish its terminal result."""

    arguments = _arguments()
    manifest = default_scaling_manifest()
    observations, environment = ANIMA_REGIONAL_LORA_ISOLATED_VRAM_RUNNER.run(manifest)
    result = evaluate_isolated_vram_result(manifest, observations)
    result_path = (
        arguments.output_root
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        / "result.json"
    )
    write_isolated_vram_result(
        result_path,
        manifest=manifest,
        result=result,
        environment=environment,
    )
    return 0 if result.passed else 1


def _arguments() -> argparse.Namespace:
    """Parse the explicit external evidence root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1")
        / "p10.1"
        / "isolated-vram",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
