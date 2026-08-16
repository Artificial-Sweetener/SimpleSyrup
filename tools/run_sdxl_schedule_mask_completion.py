# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run selected current-route SDXL schedule and mask-geometry proofs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tools.sdxl_attention_coupling_integration.selected_external_visual_runner import (
    SelectedExternalVisualRun,
)
from tools.sdxl_schedule_mask_completion.cases import (
    schedule_mask_completion_cases,
)

RUN = SelectedExternalVisualRun(
    description=__doc__ or "Run SDXL schedule and mask-geometry proofs.",
    output_root=Path(
        r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter"
        r"\ra06-schedule-mask-completion"
    ),
    log_label="SDXL schedule and mask-geometry proof",
    case_factory=schedule_mask_completion_cases,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the focused external-fixture proof family."""

    return RUN.execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
