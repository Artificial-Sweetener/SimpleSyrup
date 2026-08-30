# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the complete P6.9 tiled public-node matrix and evidence owners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.anima_attention_coupling_workflow import (
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_tiled_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_HEIGHT,
    TILE_OVERLAP,
    TILE_WIDTH,
    cases,
    select_cases,
)
from tools.anima_tiled_attention_coupling_integration.results import (
    TiledIntegrationResultRecorder,
)
from tools.anima_tiled_attention_coupling_integration.schema import (
    EXPECTED_INPUTS,
    EXPECTED_OPTIONAL_INPUTS,
    validate_public_node_metadata,
)
from tools.anima_tiled_attention_coupling_integration.workflow import (
    TiledIntegrationWorkflowBuilder,
)
from tools.comfy_api import ImageReference, JsonObject


def test_matrix_covers_modes_batches_masks_cfg_and_scheduled_primary_adapter() -> None:
    """Pin the complete managed tiled runtime matrix."""

    definitions = cases()

    assert len(definitions) == 8
    assert {case.diffusion_mode for case in definitions} == {
        "multidiffusion",
        "mixture_of_diffusers",
    }
    assert {case.tile_batch_size for case in definitions} == {1, 2, 4, 8}
    assert {case.mask_case_id for case in definitions} == {
        "overlapping-regions",
        "uncovered-center-strip",
    }
    assert {case.cfg for case in definitions} == {1.0, 4.0}
    assert all(case.regional_loras for case in definitions)
    assert any(
        len({adapter.lora_name for adapter in case.regional_loras}) == 2
        for case in definitions
    )
    assert any(
        len(case.regional_loras) == 2
        and all(adapter.schedule is not None for adapter in case.regional_loras)
        for case in definitions
    )
    assert any(case.global_loras for case in definitions)


def test_matrix_selects_an_ordered_diagnostic_subset() -> None:
    """Keep diagnostic subset execution explicit without changing full defaults."""

    selected = select_cases(("mixture_of_diffusers-batch-2", "multidiffusion-batch-1"))

    assert tuple(case.case_id for case in selected) == (
        "mixture_of_diffusers-batch-2",
        "multidiffusion-batch-1",
    )
    assert select_cases(()) == cases()
    with pytest.raises(ValueError, match="Unknown P6.9 case ids"):
        select_cases(("not-a-case",))


def test_workflows_use_public_tiled_node_and_exact_case_controls() -> None:
    """Build real Prompt Control graphs with immutable tiled controls."""

    case = cases()[5]
    workflow = TiledIntegrationWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    sampler = _sole(workflow.prompt, PUBLIC_NODE_ID)
    inputs = _inputs(sampler)

    assert _types(workflow.prompt) >= {
        PUBLIC_NODE_ID,
        "SimpleSyrup.KSamplerAttentionCoupling",
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
        "SimpleSyrupBenchmark.InstrumentModel",
        "SimpleSyrupBenchmark.ReadMetrics",
        "ImageScale",
        "VAEEncode",
    }
    assert inputs["diffusion_mode"] == case.diffusion_mode
    assert inputs["latent_tile_width"] == TILE_WIDTH
    assert inputs["latent_tile_height"] == TILE_HEIGHT
    assert inputs["latent_tile_overlap"] == TILE_OVERLAP
    assert inputs["latent_tile_batch_size"] == case.tile_batch_size
    assert inputs["denoise"] == REFINEMENT_DENOISE
    source_sampler = _sole(
        workflow.prompt,
        "SimpleSyrup.KSamplerAttentionCoupling",
    )
    assert _inputs(source_sampler)["denoise"] == 1.0
    upscale = _sole(workflow.prompt, "ImageScale")
    upscale_inputs = _inputs(upscale)
    source_decode_reference = cast(list[object], upscale_inputs["image"])
    assert workflow.prompt[str(source_decode_reference[0])]["class_type"] == "VAEDecode"
    assert upscale_inputs == {
        "image": source_decode_reference,
        "upscale_method": "lanczos",
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "crop": "disabled",
    }
    encoded_latent_id = _node_id(workflow.prompt, "VAEEncode")
    assert inputs["latent_image"] == [encoded_latent_id, 0]
    instrumented_id = _node_id(
        workflow.prompt,
        "SimpleSyrupBenchmark.InstrumentModel",
    )
    diagnostics_id = _node_id(
        workflow.prompt,
        "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
    )
    diagnostics_inputs = _inputs(workflow.prompt[diagnostics_id])
    assert diagnostics_inputs["model"] == [instrumented_id, 0]
    assert inputs["model"] == [diagnostics_id, 0]
    assert workflow.source_save_node_id is not None
    source_latent = _sole(workflow.prompt, "EmptyCosmosLatentVideo")
    assert _inputs(source_latent)["width"] == SOURCE_WIDTH
    assert _inputs(source_latent)["height"] == SOURCE_HEIGHT
    encoded = _sole(
        workflow.prompt,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
    )
    prompt = str(_inputs(encoded)["positive_prompt"])
    assert "[<lora:Anima\\style\\adapter-a.safetensors:0.8>:0.00,0.75]" in prompt
    assert "[<lora:Anima\\style\\adapter-b.safetensors:0.8>:0.25,1.00]" in prompt


def test_live_schema_validator_requires_exact_tiled_contract() -> None:
    """Pin the public ID, order, fusion options, output, and LoRA guidance."""

    metadata = _metadata()
    validate_public_node_metadata(metadata)

    cast(dict[str, object], metadata["input_order"])["required"] = ["model"]
    with pytest.raises(ValueError, match="input order"):
        validate_public_node_metadata(metadata)


def test_result_recorder_requires_complete_metrics_order_and_cleanup(
    tmp_path: Path,
) -> None:
    """Associate every labeled image once and fail closed on evidence gaps."""

    definitions = cases()
    recorder = TiledIntegrationResultRecorder(tmp_path)
    builder = TiledIntegrationWorkflowBuilder()
    for log_name in ("comfy.stdout.log", "comfy.stderr.log"):
        (tmp_path / log_name).write_text("regional LoRA tiled diagnostic\n")
    for index, case in enumerate(definitions):
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        recorder.record_case(
            case,
            workflow,
            history=_history(workflow),
            prompt_id=f"prompt-{index}",
            image_reference=ImageReference(f"{index}.png", "p69", "output"),
            image_bytes=f"png-{index}".encode(),
            source_image_reference=ImageReference(
                f"{index}-source.png",
                "p69",
                "output",
            ),
            source_image_bytes=f"source-{index}".encode(),
            wall_runtime_ms=123.0,
        )
    with pytest.raises(ValueError, match="cleanup"):
        recorder.finalize(definitions, system_stats={}, cleanup_verified=False)
    result_path = recorder.finalize(
        definitions,
        system_stats={"system": {}},
        cleanup_verified=True,
    )
    result = cast(JsonObject, json.loads(result_path.read_text(encoding="utf-8")))

    assert result["status"] == "completed"
    observations = cast(list[JsonObject], result["observations"])
    assert [item["case_id"] for item in observations] == [
        case.case_id for case in definitions
    ]
    assert observations[1]["tile_batch_size"] == 2
    assert observations[-1]["diffusion_mode"] == "mixture_of_diffusers"
    assert observations[0]["source_image_file"] == ("multidiffusion-batch-1.source.png")
    assert (tmp_path / "diagnostics.log").is_file()


def _metadata() -> JsonObject:
    """Build one exact live-schema-shaped tiled metadata fixture."""

    types: dict[str, object] = {
        "model": "MODEL",
        "positive": "CONDITIONING,CONDITIONING_BATCH",
        "negative": "CONDITIONING,CONDITIONING_BATCH",
        "latent_image": "LATENT",
    }
    required = {
        name: [types.get(name, "INT"), {"tooltip": f"{name} regional LoRA guidance"}]
        for name in EXPECTED_INPUTS
    }
    optional = {
        "region_masks": ["MASK", {"tooltip": "region_masks regional LoRA guidance"}]
    }
    required["diffusion_mode"] = [
        "COMBO",
        {
            "options": ["multidiffusion", "mixture_of_diffusers"],
            "tooltip": "diffusion mode regional LoRA guidance",
        },
    ]
    return {
        "name": PUBLIC_NODE_ID,
        "display_name": "KSampler (Attention Coupling + Tiled Diffusion)",
        "category": "SimpleSyrup/Sampling",
        "description": (
            "Regional LoRA schedule behavior with MultiDiffusion and Mixture fusion."
        ),
        "input": {"required": required, "optional": optional},
        "input_order": {
            "required": list(EXPECTED_INPUTS),
            "optional": list(EXPECTED_OPTIONAL_INPUTS),
        },
        "output": ["LATENT"],
    }


def _history(workflow: BuiltAnimaAttentionCouplingWorkflow) -> JsonObject:
    """Build one terminal history with valid tiled metrics."""

    return {
        "outputs": {
            workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.metrics_run_id,
                        "runtime_ms": 100.0,
                        "model_call_count": STEPS,
                        "peak_vram_bytes": 1024,
                    }
                ]
            }
        }
    }


def _types(prompt: dict[str, JsonObject]) -> set[str]:
    """Return all graph node class names."""

    return {str(node["class_type"]) for node in prompt.values()}


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole node with one required class."""

    matches = [node for node in prompt.values() if node["class_type"] == class_type]
    assert len(matches) == 1
    return matches[0]


def _node_id(prompt: dict[str, JsonObject], class_type: str) -> str:
    """Return the sole node identity for one required class."""

    matches = [
        node_id for node_id, node in prompt.items() if node["class_type"] == class_type
    ]
    assert len(matches) == 1
    return matches[0]


def _inputs(node: JsonObject) -> dict[str, object]:
    """Narrow one graph node's inputs."""

    return cast(dict[str, object], node["inputs"])
