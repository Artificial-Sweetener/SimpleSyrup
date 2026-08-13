# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the immutable P0.8 Prompt Control case matrix."""

from tools.prompt_control_characterization.cases import cases


def test_matrix_covers_every_required_schedule_shape() -> None:
    """Keep static, adjacent, overlap, inactive, single, and stacked cases."""

    matrix = cases()
    identifiers = [case.case_id for case in matrix]
    assert len(matrix) == 9
    assert len(set(identifiers)) == len(identifiers)
    assert {case.text_construction for case in matrix} == {
        "lazy",
        "adjacent",
        "overlapping",
        "inactive",
    }
    assert "lora-single-static" in identifiers
    assert "lora-single-scheduled" in identifiers
    assert "lora-adjacent" in identifiers
    assert "lora-stacked-overlap" in identifiers
    assert "lora-inactive" in identifiers


def test_adapter_expectations_preserve_order_and_native_keyframes() -> None:
    """Represent each WeightHook independently without merging stacked LoRAs."""

    by_id = {case.case_id: case for case in cases()}
    stacked = by_id["lora-stacked-overlap"].expected_adapters
    assert [adapter.identity for adapter in stacked] == ["adapter_a-0.4", "adapter_b-0.6"]
    assert stacked[0].keyframes == (
        (0.0, 1.0),
        (0.25, 0.0),
        (0.25, 1.0),
        (0.75, 0.0),
    )
    assert stacked[1].keyframes == (
        (0.0, 0.0),
        (0.25, 1.0),
        (0.75, 0.0),
        (0.75, 1.0),
    )
    assert by_id["lora-single-static"].expect_static_model_lora is True
    assert by_id["lora-inactive"].expected_adapters == ()
