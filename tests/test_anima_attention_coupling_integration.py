"""Verify the complete P5.9 public-node matrix and evidence owners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.anima_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID,
    cases,
)
from tools.anima_attention_coupling_integration.results import IntegrationResultRecorder
from tools.anima_attention_coupling_integration.schema import (
    EXPECTED_INPUTS,
    validate_public_node_metadata,
)
from tools.anima_attention_coupling_integration.workflow import (
    IntegrationWorkflowBuilder,
)
from tools.anima_attention_coupling_workflow import (
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject


def test_matrix_covers_counts_schedules_masks_feather_and_cfg() -> None:
    """Keep every required P5.9 behavioral dimension in the fixed matrix."""

    definitions = cases()

    assert tuple(case.regional_lora_count for case in definitions[:4]) == (0, 1, 2, 4)
    assert any(
        {adapter.schedule for adapter in case.regional_loras}
        == {(0.0, 0.5), (0.5, 1.0)}
        for case in definitions
    )
    assert any(
        case.mask_case_id == "overlapping-regions"
        and all(adapter.schedule is not None for adapter in case.regional_loras)
        for case in definitions
    )
    assert any(case.feather > 0 for case in definitions)
    assert {case.cfg for case in definitions} == {1.0, 4.0}
    split = definitions[2]
    assert len(split.global_loras) == 1
    assert "GLOBAL_ADAPTER" in split.global_loras[0].lora_name
    assert [adapter.region_index for adapter in split.regional_loras] == [0, 1]
    assert "ADAPTER_A" in split.regional_loras[0].lora_name
    assert "ADAPTER_B" in split.regional_loras[1].lora_name


def test_workflows_use_public_node_and_shared_sampler_owners() -> None:
    """Build the public sampler graph without the benchmark profile shortcut."""

    definitions = cases()
    builder = _builder()
    workflow = builder.build(
        definitions[0],
        run_id="run",
        mask_names=("left.png", "right.png"),
    )

    assert _types(workflow.prompt) >= {
        PUBLIC_NODE_ID,
        "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
        "SimpleSyrupBenchmark.InstrumentModel",
        "SimpleSyrupBenchmark.ReadMetrics",
    }
    assert "SimpleSyrupBenchmark.StaticAnimaRegionalProfile" not in _types(
        workflow.prompt
    )
    sampler = _sole(workflow.prompt, PUBLIC_NODE_ID)
    assert _inputs(sampler)["region_mask_feather"] == 0
    assert _inputs(sampler)["cfg"] == 1.0


def test_live_schema_validator_requires_exact_contract_and_lora_guidance() -> None:
    """Pin the public ID, order, types, output, description, and tooltips."""

    metadata = _metadata()
    validate_public_node_metadata(metadata)

    cast(dict[str, object], metadata["input_order"])["required"] = ["model"]
    with pytest.raises(ValueError, match="input order"):
        validate_public_node_metadata(metadata)


def test_result_recorder_rejects_incomplete_metrics_and_requires_cleanup(
    tmp_path: Path,
) -> None:
    """Associate each image once and fail closed on evidence or cleanup gaps."""

    recorder = IntegrationResultRecorder(tmp_path)
    definitions = cases()
    builder = _builder()
    for log_name in ("comfy.stdout.log", "comfy.stderr.log"):
        (tmp_path / log_name).write_text("regional LoRA diagnostic\n", encoding="utf-8")
    for index, case in enumerate(definitions):
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        history = _history(workflow)
        recorder.record_case(
            case,
            workflow,
            history=history,
            prompt_id=f"prompt-{index}",
            image_reference=ImageReference(f"{index}.png", "p59", "output"),
            image_bytes=f"png-{index}".encode(),
            wall_runtime_ms=123.0,
        )
    with pytest.raises(ValueError, match="cleanup"):
        recorder.finalize(definitions, system_stats={}, cleanup_verified=False)
    result_path = recorder.finalize(
        definitions, system_stats={"system": {}}, cleanup_verified=True
    )
    result = cast(JsonObject, json.loads(result_path.read_text(encoding="utf-8")))
    assert result["status"] == "completed"
    observations = cast(list[JsonObject], result["observations"])
    assert [item["case_id"] for item in observations] == [
        case.case_id for case in definitions
    ]
    assert observations[2]["regional_lora_count"] == 2
    assert cast(JsonObject, observations[2]["metrics"])["model_call_count"] == STEPS
    assert (tmp_path / "diagnostics.log").is_file()

    bad_workflow = builder.build(
        definitions[0], run_id="bad", mask_names=("left.png", "right.png")
    )
    bad_history = _history(bad_workflow, call_count=STEPS - 1)
    with pytest.raises(ValueError, match="complete denoiser"):
        IntegrationResultRecorder(tmp_path).record_case(
            definitions[0],
            bad_workflow,
            history=bad_history,
            prompt_id="bad",
            image_reference=ImageReference("bad.png", "p59", "output"),
            image_bytes=b"bad",
            wall_runtime_ms=1.0,
        )


def _metadata() -> JsonObject:
    """Build one exact live-schema-shaped metadata fixture."""

    types = {
        "model": "MODEL",
        "positive": "CONDITIONING,CONDITIONING_BATCH",
        "negative": "CONDITIONING,CONDITIONING_BATCH",
        "region_masks": "MASK",
        "latent_image": "LATENT",
    }
    required = {
        name: [types.get(name, "INT"), {"tooltip": f"{name} regional LoRA guidance"}]
        for name in EXPECTED_INPUTS
    }
    return {
        "name": PUBLIC_NODE_ID,
        "display_name": "KSampler (Attention Coupling)",
        "category": "SimpleSyrup/Sampling",
        "description": "Regional LoRA schedule and overlap behavior.",
        "input": {"required": required},
        "input_order": {"required": list(EXPECTED_INPUTS)},
        "output": ["LATENT"],
    }


def _history(
    workflow: BuiltAnimaAttentionCouplingWorkflow, *, call_count: int = STEPS
) -> JsonObject:
    """Build one terminal history with valid benchmark metrics."""

    return {
        "outputs": {
            workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.metrics_run_id,
                        "runtime_ms": 100.0,
                        "model_call_count": call_count,
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


def _inputs(node: JsonObject) -> dict[str, object]:
    """Narrow one graph node's inputs."""

    return cast(dict[str, object], node["inputs"])


def _builder() -> IntegrationWorkflowBuilder:
    """Return the P5.9 workflow coordinator."""

    return IntegrationWorkflowBuilder()
