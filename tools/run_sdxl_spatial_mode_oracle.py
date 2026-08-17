# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the full-strength SDXL full, tiled, and Contextual oracle."""

from __future__ import annotations

from collections.abc import Sequence

from tools.comfy_integration.default_paths import default_benchmark_artifact_root
from tools.sdxl_attention_coupling_integration.selected_external_visual_runner import (
    SelectedExternalVisualRun,
)
from tools.sdxl_spatial_mode_oracle import spatial_mode_oracle_cases

RUN = SelectedExternalVisualRun(
    description=__doc__ or "Run the SDXL spatial-mode oracle.",
    output_root=default_benchmark_artifact_root(
        "universal-regional-adapter/sdxl-spatial-mode-oracle"
    ),
    log_label="SDXL spatial-mode oracle",
    case_factory=spatial_mode_oracle_cases,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the explicitly selected spatial-mode oracle case."""

    return RUN.execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
