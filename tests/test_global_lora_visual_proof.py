"""Verify the fixed global and regional ADAPTER_A visual proof matrix."""

from __future__ import annotations

from tools.global_lora_visual_proof.matrix import cases, render_positive_prompt


def test_matrix_covers_global_strength_regional_and_duplicate_placement() -> None:
    """Keep every requested placement explicit and ordered."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "no-lora",
        "global-adapter_a-050",
        "global-adapter_a-100",
        "regional-adapter_a-100-left",
        "duplicate-global-regional-adapter_a-100",
    ]
    assert [case.global_strength for case in definitions] == [None, 0.5, 1.0, None, 1.0]
    assert [case.regional_strength for case in definitions] == [
        None,
        None,
        None,
        1.0,
        1.0,
    ]
    assert [case.expect_overlap_rejection for case in definitions] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_prompt_renderer_places_adapter_a_only_in_declared_segments() -> None:
    """Remove prior tags before rendering exact global and left-region uses."""

    source = (
        "global text[SEP]left text "
        r"<lora:Anima\style\adapter-a.safetensors:0.8>"
        "[SEP]right text"
    )

    rendered = [render_positive_prompt(source, case).split("[SEP]") for case in cases()]

    assert all("<lora:" not in segment for segment in rendered[0])
    assert all("<lora:" not in segment for segment in rendered[1])
    assert all("<lora:" not in segment for segment in rendered[2])
    assert "adapter-a.safetensors:1" in rendered[3][1]
    assert "<lora:" not in rendered[3][0] + rendered[3][2]
    assert "<lora:" not in rendered[4][0]
    assert "adapter-a.safetensors:1" in rendered[4][1]
