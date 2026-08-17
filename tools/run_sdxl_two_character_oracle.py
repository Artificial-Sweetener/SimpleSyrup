# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the accepted SDXL two-character regional-LoRA oracle."""

from __future__ import annotations

from collections.abc import Sequence

from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
)
from tools.sdxl_attention_coupling_integration.selected_external_visual_runner import (
    SelectedExternalVisualRun,
)
from tools.sdxl_two_character_oracle import two_character_oracle_cases

RUN = SelectedExternalVisualRun(
    description=__doc__ or "Run the SDXL two-character oracle.",
    output_root=default_benchmark_artifact_root(
        "universal-regional-adapter/sdxl-two-character-oracle"
    ),
    log_label="SDXL two-character oracle",
    case_factory=two_character_oracle_cases,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the explicitly selected oracle case."""

    return RUN.execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
