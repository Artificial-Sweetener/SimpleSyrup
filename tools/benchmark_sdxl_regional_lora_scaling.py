# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark warmed zero/one/four SDXL regional adapter uses."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_prompt_fixture import (
    load_visual_prompt_set,
)
from tools.sdxl_regional_lora_performance.scaling_runner import (
    run_active_use_scaling,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "universal-regional-adapter/sdxl-active-use-scaling"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load external fixtures and run the active-use scaling matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=default_comfy_root())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = run_active_use_scaling(
            artifacts,
            inventory=SdxlVisualInventory.load(args.inventory),
            prompts=load_visual_prompt_set(args.prompt_case),
            comfy_root=args.comfy_root,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("SDXL active-use scaling failed at %s", artifacts.root)
        return 1
    LOGGER.info("SDXL active-use scaling completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
