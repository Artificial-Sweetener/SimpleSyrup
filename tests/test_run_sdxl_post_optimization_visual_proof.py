# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify selected post-optimization visual CLI delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools import run_sdxl_post_optimization_visual_proof
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)


def test_post_optimization_cli_executes_only_selected_case(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Pass one explicit case through the shared selector to the executor."""

    inventory_path = tmp_path / "inventory.json"
    prompts_path = tmp_path / "prompts.json"
    inventory = object()
    prompts = object()
    first = _Case("first")
    second = _Case("second")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        SdxlVisualInventory,
        "load",
        lambda path: inventory if path == inventory_path else None,
    )
    monkeypatch.setattr(
        run_sdxl_post_optimization_visual_proof,
        "load_visual_prompt_set",
        lambda path: prompts if path == prompts_path else None,
    )
    monkeypatch.setattr(
        run_sdxl_post_optimization_visual_proof,
        "post_optimization_visual_cases",
        lambda supplied_inventory, supplied_prompts: (
            (
                first,
                second,
            )
            if (supplied_inventory, supplied_prompts) == (inventory, prompts)
            else ()
        ),
    )

    def execute(
        _artifacts: object,
        *,
        inventory: object,
        cases: tuple[object, ...],
        comfy_root: Path,
        readiness_timeout: float,
        prompt_timeout: float,
        seed: int,
    ) -> Path:
        calls.append(
            (
                inventory,
                cases,
                comfy_root,
                readiness_timeout,
                prompt_timeout,
                seed,
            )
        )
        return tmp_path / "result.json"

    monkeypatch.setattr(
        run_sdxl_post_optimization_visual_proof,
        "execute_visual_cases",
        execute,
    )

    result = run_sdxl_post_optimization_visual_proof.main(
        (
            "--inventory",
            str(inventory_path),
            "--prompt-case",
            str(prompts_path),
            "--comfy-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "artifacts"),
            "--case-id",
            "second",
            "--seed",
            "7429113058",
        )
    )

    assert result == 0
    assert calls == [(inventory, (second,), tmp_path, 240.0, 1200.0, 7429113058)]


class _Case:
    """Expose only the case identity required by the shared selector."""

    def __init__(self, case_id: str) -> None:
        """Retain one stable case identity."""

        self.case_id = case_id
