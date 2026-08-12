"""Verify the complete P7.8 combined public-node evidence owners."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from tools.anima_contextual_attention_coupling_integration.matrix import (
    GLOBAL_STEPS,
    PUBLIC_NODE_ID,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    SUBTOKEN_MASK_CASE_ID,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    ContextualIntegrationCase,
    cases,
    select_cases,
    subtoken_mask_case,
)
from tools.anima_contextual_attention_coupling_integration.results import (
    ContextualIntegrationResultRecorder,
)
from tools.anima_contextual_attention_coupling_integration.schema import (
    EXPECTED_OPTIONAL_INPUTS,
    EXPECTED_REQUIRED_INPUTS,
    validate_public_node_metadata,
)
from tools.anima_contextual_attention_coupling_integration.workflow import (
    ContextualIntegrationWorkflowBuilder,
)
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_api import ImageReference, JsonObject


def test_matrix_covers_modes_branches_segs_schedules_and_subtoken_masks(
    tmp_path: Path,
) -> None:
    """Pin every required managed Contextual runtime dimension."""

    definitions = cases()

    assert len(definitions) == 6
    assert {case.diffusion_mode for case in definitions} == {
        "multidiffusion",
        "mixture_of_diffusers",
    }
    assert {case.branch_mode for case in definitions} == {
        "local",
        "reduced_global",
        "sub_token_reduced_global",
    }
    assert {case.global_steps for case in definitions} == {0, GLOBAL_STEPS}
    assert any(case.use_segs for case in definitions)
    assert all(
        len(case.regional_loras) == 2
        and all(adapter.schedule is not None for adapter in case.regional_loras)
        for case in definitions
    )
    assert all(
        any("ADAPTER_A" in adapter.lora_name for adapter in case.regional_loras)
        for case in definitions
    )
    writer = MaskArtifactWriter(tmp_path, "p78-test")
    names = writer.write_case(
        subtoken_mask_case(),
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    with Image.open(tmp_path / names[0]) as image:
        active = sum(1 for pixel in image.get_flattened_data() if pixel != (0, 0, 0))
    assert active == TARGET_HEIGHT


def test_matrix_selects_an_ordered_diagnostic_subset() -> None:
    """Keep diagnostic subset execution explicit without changing full defaults."""

    selected = select_cases(
        (
            "mixture_of_diffusers-sub-token-global",
            "multidiffusion-local-segs",
        )
    )

    assert tuple(case.case_id for case in selected) == (
        "mixture_of_diffusers-sub-token-global",
        "multidiffusion-local-segs",
    )
    assert select_cases(()) == cases()
    with pytest.raises(ValueError, match="Unknown P7.8 case ids"):
        select_cases(("not-a-case",))


def test_workflows_use_combined_node_segs_and_exact_contextual_controls() -> None:
    """Build real Prompt Control graphs with optional graph-owned SEGS."""

    builder = ContextualIntegrationWorkflowBuilder()
    local_case = cases()[0]
    global_case = cases()[1]
    local = builder.build(
        local_case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    reduced_global = builder.build(
        global_case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    local_sampler = _sole(local.prompt, PUBLIC_NODE_ID)
    local_inputs = _inputs(local_sampler)
    global_inputs = _inputs(_sole(reduced_global.prompt, PUBLIC_NODE_ID))

    assert _types(local.prompt) >= {
        PUBLIC_NODE_ID,
        "MaskToImage",
        "ImageFromBatch",
        "ImageToMask",
        "SimpleSyrup.MaskToSEGS",
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
        "SimpleSyrupBenchmark.InstrumentModel",
        "SimpleSyrupBenchmark.ReadMetrics",
        "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
        "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
    }
    assert local_inputs["diffusion_mode"] == "multidiffusion"
    assert local_inputs["global_steps"] == 0
    assert local_inputs["denoise"] == REFINEMENT_DENOISE
    assert global_inputs["global_steps"] == GLOBAL_STEPS
    segs_reference = cast(list[object], local_inputs["segs"])
    assert local.prompt[str(segs_reference[0])]["class_type"] == (
        "SimpleSyrup.MaskToSEGS"
    )
    encoded = _sole(
        local.prompt, "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
    )
    prompt = str(_inputs(encoded)["positive_prompt"])
    assert "[<lora:Anima\\style\\adapter-a.safetensors:0.8>:0.00,0.75]" in prompt
    assert "[<lora:Anima\\style\\adapter-b.safetensors:0.8>:0.25,1.00]" in prompt
    assert "SimpleSyrup.MaskToSEGS" not in _types(
        builder.build(
            cases()[2],
            run_id="run",
            mask_names=("pixel.png", "rest.png"),
        ).prompt
    )
    assert _types(local.prompt) >= {
        "SimpleSyrup.KSamplerAttentionCoupling",
        "ImageScale",
        "VAEEncode",
    }
    source_latent = _sole(local.prompt, "EmptyCosmosLatentVideo")
    assert _inputs(source_latent)["width"] == SOURCE_WIDTH
    assert _inputs(source_latent)["height"] == SOURCE_HEIGHT
    upscale = _sole(local.prompt, "ImageScale")
    assert _inputs(upscale)["width"] == TARGET_WIDTH
    assert _inputs(upscale)["height"] == TARGET_HEIGHT
    assert local.source_save_node_id is not None


def test_live_schema_validator_requires_exact_combined_contract() -> None:
    """Pin public identity, inputs, outputs, fusion options, and guidance."""

    metadata = _metadata()
    validate_public_node_metadata(metadata)

    cast(dict[str, object], metadata["input_order"])["required"] = ["model"]
    with pytest.raises(ValueError, match="input order"):
        validate_public_node_metadata(metadata)


def test_result_recorder_proves_branch_calls_order_and_cleanup(tmp_path: Path) -> None:
    """Associate labeled images and prove reduced-global calls were measured."""

    definitions = cases()
    recorder = ContextualIntegrationResultRecorder(tmp_path)
    builder = ContextualIntegrationWorkflowBuilder()
    for log_name in ("comfy.stdout.log", "comfy.stderr.log"):
        (tmp_path / log_name).write_text(
            "Contextual regional LoRA Attention Coupling diagnostic\n",
            encoding="utf-8",
        )
    image_bytes = _png_bytes(TARGET_WIDTH, TARGET_HEIGHT)
    source_image_bytes = _png_bytes(SOURCE_WIDTH, SOURCE_HEIGHT)
    for index, case in enumerate(definitions):
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        model_calls = 24 if case.branch_mode == "local" else 27
        history: JsonObject = {
            "outputs": {
                workflow.metrics_node_id: {
                    "benchmark_metrics": [
                        {
                            "run_id": workflow.metrics_run_id,
                            "model_call_count": model_calls,
                            "runtime_ms": 1000.0 + index,
                            "peak_vram_bytes": 1,
                        }
                    ]
                },
                workflow.diagnostics_node_id: {
                    "regional_diagnostics": [
                        _diagnostics(workflow.diagnostics_run_id, case)
                    ]
                },
            }
        }
        recorder.record_case(
            case,
            workflow,
            history=history,
            prompt_id=f"prompt-{index}",
            image_reference=ImageReference(f"{case.case_id}.png", "p7.8", "output"),
            image_bytes=image_bytes,
            source_image_reference=ImageReference(
                f"{case.case_id}.source.png", "p7.8", "output"
            ),
            source_image_bytes=source_image_bytes,
            wall_runtime_ms=1200.0 + index,
        )

    result_path = recorder.finalize(
        definitions,
        system_stats={"device": "test"},
        cleanup_verified=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert [item["case_id"] for item in result["observations"]] == [
        case.case_id for case in definitions
    ]
    subtoken = next(
        item
        for item in result["observations"]
        if item["mask_case_id"] == SUBTOKEN_MASK_CASE_ID
    )
    assert subtoken["subtoken_authored_width_pixels"] == 1
    assert subtoken["declared_regional_adapter_executions"] == 2


def test_result_recorder_retains_raw_sidecars_before_semantic_rejection(
    tmp_path: Path,
) -> None:
    """Preserve exact workflow history when diagnostic validation fails."""

    case = cases()[0]
    workflow = ContextualIntegrationWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    history: JsonObject = {"outputs": {}}
    recorder = ContextualIntegrationResultRecorder(tmp_path)

    with pytest.raises(ValueError, match="missing metrics output"):
        recorder.record_case(
            case,
            workflow,
            history=history,
            prompt_id="prompt",
            image_reference=ImageReference("image.png", "p7.8", "output"),
            image_bytes=_png_bytes(TARGET_WIDTH, TARGET_HEIGHT),
            source_image_reference=ImageReference("source.png", "p7.8", "output"),
            source_image_bytes=_png_bytes(SOURCE_WIDTH, SOURCE_HEIGHT),
            wall_runtime_ms=1.0,
        )

    assert (tmp_path / f"{case.case_id}.history.json").is_file()
    assert (tmp_path / f"{case.case_id}.workflow.json").is_file()


def _metadata() -> JsonObject:
    """Return one valid live-style combined-node metadata object."""

    required: JsonObject = {
        name: ["INT", {"tooltip": "setting"}] for name in EXPECTED_REQUIRED_INPUTS
    }
    required.update(
        {
            "model": ["MODEL", {"tooltip": "global LoRA model"}],
            "positive": [
                "CONDITIONING,CONDITIONING_BATCH",
                {"tooltip": "regional LoRA positive"},
            ],
            "negative": [
                "CONDITIONING,CONDITIONING_BATCH",
                {"tooltip": "regional LoRA negative"},
            ],
            "region_masks": ["MASK", {"tooltip": "regional LoRA masks"}],
            "latent_image": ["LATENT", {"tooltip": "latent"}],
            "diffusion_mode": [
                "COMBO",
                {
                    "options": ["multidiffusion", "mixture_of_diffusers"],
                    "tooltip": "fusion",
                },
            ],
        }
    )
    return {
        "name": PUBLIC_NODE_ID,
        "display_name": "KSampler (Attention Coupling + Contextual Diffusion)",
        "category": "SimpleSyrup/Sampling",
        "input_order": {
            "required": list(EXPECTED_REQUIRED_INPUTS),
            "optional": list(EXPECTED_OPTIONAL_INPUTS),
        },
        "input": {
            "required": required,
            "optional": {"segs": ["SEGS", {"tooltip": "SEGS guidance"}]},
        },
        "output": ["LATENT", "SEGS"],
        "description": (
            "LoRA schedules on local and reduced-global views with optional SEGS"
        ),
    }


def _png_bytes(width: int, height: int) -> bytes:
    """Return one valid fixed-size test PNG."""

    stream = BytesIO()
    Image.new("RGB", (width, height), color=(0, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def _diagnostics(
    run_id: str,
    case: ContextualIntegrationCase,
) -> JsonObject:
    """Return one complete live-style structured diagnostics result."""

    snapshots: list[object] = [_diagnostic_snapshot("tile")]
    if case.global_steps > 0:
        snapshots.append(_diagnostic_snapshot("contextual_global"))
    return {
        "run_id": run_id,
        "record_count": len(snapshots),
        "snapshots": snapshots,
    }


def _diagnostic_snapshot(spatial_mode: str) -> JsonObject:
    """Return one structured local or reduced-global snapshot."""

    return {
        "spatial_mode": spatial_mode,
        "active_region_indices": [0, 1],
        "canonical_mask": {
            "regions": [
                {"region_index": 0, "nonzero_fraction": 0.5},
                {"region_index": 1, "nonzero_fraction": 0.5},
            ]
        },
        "query_grid": {"token_count": 64},
        "adapter_uses": [
            {"region_index": 0, "active": True},
            {"region_index": 1, "active": True},
        ],
    }


def _types(prompt: dict[str, JsonObject]) -> set[str]:
    """Return graph class types."""

    return {str(node["class_type"]) for node in prompt.values()}


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole graph node with one class type."""

    matches = [node for node in prompt.values() if node["class_type"] == class_type]
    assert len(matches) == 1
    return matches[0]


def _inputs(node: JsonObject) -> JsonObject:
    """Return one graph node's narrowed inputs."""

    value = node["inputs"]
    assert isinstance(value, dict)
    return cast(JsonObject, value)
