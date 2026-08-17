# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run selected unresolved SDXL composed-LoRA visual proofs."""

from __future__ import annotations

from collections.abc import Sequence

from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
)
from tools.sdxl_attention_coupling_integration.selected_external_visual_runner import (
    SelectedExternalVisualRun,
)
from tools.sdxl_composed_lora_completion.cases import (
    composed_lora_completion_cases,
)

DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "universal-regional-adapter/composed-lora-completion"
)
RUN = SelectedExternalVisualRun(
    description=__doc__ or "Run SDXL composed-LoRA visual proofs.",
    output_root=DEFAULT_OUTPUT_ROOT,
    log_label="Composed SDXL LoRA proof",
    case_factory=composed_lora_completion_cases,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the focused external-fixture proof family."""

    return RUN.execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
