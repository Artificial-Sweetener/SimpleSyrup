# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the U11 native-SDXL regional-LoRA visual acceptance matrix."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.visual_case_selection import (
    select_visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    execute_visual_cases,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\u11"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        inventory = SdxlVisualInventory.load(args.inventory)
        cases = select_visual_cases(visual_cases(inventory), tuple(args.case_id))
        result = execute_visual_cases(
            artifacts,
            inventory=inventory,
            cases=cases,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Managed U11 SDXL visual matrix failed at %s", artifacts.root)
        return 1
    LOGGER.info("Managed U11 SDXL visual matrix completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
