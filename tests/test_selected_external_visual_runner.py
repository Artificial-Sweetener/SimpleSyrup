# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the shared selected external-fixture visual runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration import (
    selected_external_visual_runner as runner_module,
)
from tools.sdxl_attention_coupling_integration.selected_external_visual_runner import (
    SelectedExternalVisualRun,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)


def test_runner_loads_fixtures_and_preserves_explicit_case_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate one selected family to the managed executor without policy drift."""

    inventory = visual_inventory(tmp_path)
    prompts = _prompts()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        SdxlVisualInventory,
        "load",
        staticmethod(lambda path: inventory),
    )
    monkeypatch.setattr(
        runner_module,
        "load_visual_prompt_set",
        lambda path: prompts,
    )

    def execute(
        artifacts: IntegrationArtifacts,
        *,
        inventory: SdxlVisualInventory,
        cases: tuple[SdxlVisualCase, ...],
        comfy_root: Path,
        readiness_timeout: float,
        prompt_timeout: float,
    ) -> Path:
        """Capture the exact selected execution contract."""

        observed.update(
            artifacts=artifacts,
            inventory=inventory,
            cases=cases,
            comfy_root=comfy_root,
            readiness_timeout=readiness_timeout,
            prompt_timeout=prompt_timeout,
        )
        return artifacts.root / "result.json"

    monkeypatch.setattr(runner_module, "execute_visual_cases", execute)
    run = SelectedExternalVisualRun(
        "Focused proof",
        tmp_path / "default-output",
        "Focused proof",
        _cases,
    )
    output_root = tmp_path / "selected-output"

    status = run.execute(
        (
            "--inventory",
            str(tmp_path / "inventory.json"),
            "--prompt-case",
            str(tmp_path / "prompts.json"),
            "--case-id",
            "second",
            "--case-id",
            "first",
            "--comfy-root",
            str(tmp_path / "comfy"),
            "--output-root",
            str(output_root),
            "--readiness-timeout",
            "12",
            "--prompt-timeout",
            "34",
        )
    )

    assert status == 0
    assert observed["inventory"] is inventory
    observed_cases = observed["cases"]
    assert isinstance(observed_cases, tuple)
    assert all(isinstance(case, SdxlVisualCase) for case in observed_cases)
    assert tuple(case.case_id for case in observed_cases) == ("second", "first")
    assert observed["comfy_root"] == tmp_path / "comfy"
    assert observed["readiness_timeout"] == 12.0
    assert observed["prompt_timeout"] == 34.0
    artifacts = observed["artifacts"]
    assert isinstance(artifacts, IntegrationArtifacts)
    assert artifacts.root.parent == output_root


def _cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return two anonymous cases while consuming both external fixtures."""

    assert inventory.checkpoint_label
    assert prompts.base_positive_g
    return (
        SdxlVisualCase("first", "First"),
        SdxlVisualCase("second", "Second"),
    )


def _prompts() -> SdxlVisualPromptSet:
    """Return one anonymous complete prompt fixture."""

    return SdxlVisualPromptSet(
        base_positive_g="global positive g",
        base_positive_l="global positive l",
        base_negative_g="global negative g",
        base_negative_l="global negative l",
        left_positive_g="left positive g",
        left_positive_l="left positive l",
        right_positive_g="right positive g",
        right_positive_l="right positive l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )
