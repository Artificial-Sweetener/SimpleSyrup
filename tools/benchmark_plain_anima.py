# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run a fresh native-Anima P5.7-compatible performance comparison."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from tools.anima_regional_lora_performance.artifacts import load_performance_artifacts
from tools.anima_regional_lora_performance.manifest import default_manifest
from tools.anima_regional_lora_performance.plain_comparison import (
    run_plain_anima_comparison,
    write_plain_anima_result,
)


def main() -> int:
    """Execute the native comparator and print its median evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = default_manifest(
        repeats=3,
        artifacts=load_performance_artifacts(arguments.artifact_inventory),
    )
    result, environment = run_plain_anima_comparison(manifest)
    output = (
        arguments.output_root
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        / "result.json"
    )
    write_plain_anima_result(output, result=result, environment=environment)
    median_runtime_ms = cast(float, result["median_runtime_ms"])
    median_peak_vram_bytes = cast(int, result["median_peak_vram_bytes"])
    print(
        f"plain-anima: median={median_runtime_ms:.2f} ms, "
        f"peak_vram={median_peak_vram_bytes} bytes"
    )
    print(f"result={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
