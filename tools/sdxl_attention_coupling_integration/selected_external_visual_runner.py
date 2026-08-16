# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run explicitly selected SDXL visual cases from external prompt fixtures."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from tools.comfy_integration.artifacts import IntegrationArtifacts

from .visual_case_model import SdxlVisualCase
from .visual_case_selection import select_visual_cases
from .visual_inventory import SdxlVisualInventory
from .visual_lora_baseline_cases import SdxlVisualPromptSet
from .visual_matrix_execution import execute_visual_cases
from .visual_prompt_fixture import load_visual_prompt_set

LOGGER = logging.getLogger(__name__)

ExternalVisualCaseFactory: TypeAlias = Callable[
    [SdxlVisualInventory, SdxlVisualPromptSet],
    tuple[SdxlVisualCase, ...],
]


@dataclass(frozen=True, slots=True)
class SelectedExternalVisualRun:
    """Own one focused case family's external-fixture CLI lifecycle."""

    description: str
    output_root: Path
    log_label: str
    case_factory: ExternalVisualCaseFactory

    def execute(self, argv: Sequence[str] | None = None) -> int:
        """Load fixtures, select explicit cases, and execute one managed run."""

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("--inventory", type=Path, required=True)
        parser.add_argument("--prompt-case", type=Path, required=True)
        parser.add_argument("--case-id", action="append", required=True)
        parser.add_argument("--comfy-root", type=Path, default=Path(r"<COMFY_ROOT>"))
        parser.add_argument("--output-root", type=Path, default=self.output_root)
        parser.add_argument("--readiness-timeout", type=float, default=240.0)
        parser.add_argument("--prompt-timeout", type=float, default=1200.0)
        args = parser.parse_args(argv)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        artifacts = IntegrationArtifacts(args.output_root)
        try:
            inventory = SdxlVisualInventory.load(args.inventory)
            prompts = load_visual_prompt_set(args.prompt_case)
            cases = select_visual_cases(
                self.case_factory(inventory, prompts),
                tuple(args.case_id),
            )
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
            LOGGER.exception("%s failed at %s", self.log_label, artifacts.root)
            return 1
        LOGGER.info("%s completed: %s", self.log_label, result)
        return 0
