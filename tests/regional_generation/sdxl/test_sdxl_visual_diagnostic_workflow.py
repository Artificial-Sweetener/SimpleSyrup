# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact image-free projection of one SDXL visual workflow."""

from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_diagnostic_workflow import (
    build_sdxl_visual_diagnostic_workflow,
)


def test_diagnostic_projection_retains_exact_sampler_and_terminals() -> None:
    """Remove image work while preserving the locked regional graph."""

    built = build_sdxl_visual_diagnostic_workflow(
        run_id="generic-run",
        checkpoint_name="generic-checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=_case(),
    )
    class_types = tuple(node["class_type"] for node in built.prompt.values())

    assert class_types.count("SimpleSyrup.KSamplerAttentionCoupling") == 1
    assert class_types.count("SimpleSyrupBenchmark.InstrumentModel") == 1
    assert class_types.count("SimpleSyrupBenchmark.ReadMetrics") == 1
    assert class_types.count("SimpleSyrupBenchmark.CaptureRegionalDiagnostics") == 1
    assert class_types.count("SimpleSyrupBenchmark.ReadRegionalDiagnostics") == 1
    assert "VAEDecode" not in class_types
    assert "SaveImage" not in class_types
    assert built.prompt[built.metrics_node_id]["class_type"] == (
        "SimpleSyrupBenchmark.ReadMetrics"
    )
    assert built.prompt[built.diagnostics_node_id]["class_type"] == (
        "SimpleSyrupBenchmark.ReadRegionalDiagnostics"
    )


def _case() -> SdxlVisualCase:
    """Return one generic left-only full-strength declaration."""

    return SdxlVisualCase(
        "generic-left-only",
        "Generic left only",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        base_negative_g="bad quality",
        base_negative_l="bad quality",
        left_g="left subject",
        left_l="left subject",
        right_g="right subject",
        right_l="right subject",
        left_negative_g="wrong left",
        left_negative_l="wrong left",
        right_negative_g="wrong right",
        right_negative_l="wrong right",
        regional_prompt_weight=1.0,
        left_adapters=(
            RegionalVisualAdapter(
                "generic-left.safetensors",
                1.0,
                1.0,
                ((0.0, 1.0),),
            ),
        ),
    )
