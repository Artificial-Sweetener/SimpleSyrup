# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fixed input selection for the thin P0.7 coordinator."""

from __future__ import annotations

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.characterize_anima_lora import _positive_prompt


def test_cli_uses_the_single_global_only_manifest_prompt() -> None:
    """Keep regional conditioning semantics out of the global LoRA baseline."""

    manifest = load_manifest()

    assert _positive_prompt(manifest) == next(
        case.global_prompt for case in manifest.cases if not case.regional_prompts
    )
