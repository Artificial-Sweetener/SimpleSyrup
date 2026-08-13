# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the immutable U1 Anima regression-oracle definition."""

from pathlib import Path

from tools.anima_regression_oracle.manifest import default_manifest


def test_manifest_pins_visual_managed_performance_and_gate_contracts() -> None:
    """Keep accepted evidence and executable gates complete and ordered."""

    manifest = default_manifest(Path.cwd())

    assert [image.identity for image in manifest.images] == [
        "character_a-phased-composition",
        "adapter_a-one-side-phased-composition",
        "global-style-character-control",
        "global-adapter_a-050-regional-character_a",
        "global-adapter_a-100-regional-character_a",
    ]
    assert [evidence.observation_count for evidence in manifest.managed_results] == [
        None,
        None,
        3,
        8,
        8,
        6,
        1,
    ]
    assert manifest.performance.adapter_target_count == 448
    assert manifest.performance.profile_limits == (
        ("attention-only", 0.0),
        ("regional-lora-1", 15.0),
        ("regional-lora-4", 35.0),
    )
    assert manifest.focused_command.arguments[1:5] == (
        "-m",
        "pytest",
        "-n",
        "auto",
    )
    assert [command.identity for command in manifest.repository_commands] == [
        "ruff-format-check",
        "ruff-check",
        "strict-mypy",
        "full-python-suite",
    ]
    assert [command.identity for command in manifest.managed_rerun_commands] == [
        "character_a-managed-proof",
        "adapter_a-managed-proof",
        "global-style-character-proof",
        "full-managed-matrix",
        "tiled-managed-matrix",
        "contextual-managed-matrix",
        "performance-gate",
    ]
    assert all(
        command.arguments[1] == "-m" for command in manifest.managed_rerun_commands
    )


def test_manifest_names_focused_tests_explicitly() -> None:
    """Prevent broad filename discovery from silently changing U1 coverage."""

    arguments = default_manifest(Path.cwd()).focused_command.arguments
    tests = tuple(argument for argument in arguments if argument.startswith("tests/"))

    assert len(tests) == 32
    assert "tests/test_anima_module_surface.py" in tests
    assert "tests/test_anima_multi_lora_fidelity.py" in tests
    assert "tests/test_anima_composition_phase.py" in tests
    assert "tests/test_anima_tiled_attention_coupling_integration.py" in tests
    assert "tests/test_anima_contextual_attention_coupling_integration.py" in tests
    assert "tests/test_anima_regional_lora_performance_results.py" in tests
